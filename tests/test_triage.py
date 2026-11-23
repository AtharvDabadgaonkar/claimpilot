"""Triage scoring: bands, reason codes and red flags."""

from __future__ import annotations

from claimpilot.coverage import CoverageStatus
from claimpilot.triage import (
    Band,
    red_flags,
    score_complexity,
    score_fraud,
    score_severity,
    triage_claim,
)

SIMPLE_CLAIM = {
    "claim_id": "TEST-001",
    "total_claim_amount": 4_000,
    "incident_severity": "Minor Damage",
    "has_injury": False,
    "bodily_injuries": 0,
    "multi_party": False,
    "number_of_vehicles_involved": 1,
    "witnesses": 1,
    "missing_police_report": False,
    "missing_property_damage_info": False,
    "incident_hour_of_the_day": 14,
    "months_as_customer": 120,
    "policy_annual_premium": 1200,
}

COMPLEX_CLAIM = {
    **SIMPLE_CLAIM,
    "claim_id": "TEST-002",
    "total_claim_amount": 95_000,
    "incident_severity": "Total Loss",
    "has_injury": True,
    "bodily_injuries": 2,
    "multi_party": True,
    "number_of_vehicles_involved": 3,
    "witnesses": 0,
    "missing_police_report": True,
}


# --- Severity -----------------------------------------------------------------


def test_severity_bands_follow_the_amount():
    assert score_severity({**SIMPLE_CLAIM, "total_claim_amount": 5_000}).band is Band.LOW
    assert score_severity({**SIMPLE_CLAIM, "total_claim_amount": 40_000}).band is Band.MEDIUM
    assert score_severity({**SIMPLE_CLAIM, "total_claim_amount": 90_000}).band is Band.HIGH


def test_total_loss_forces_high_severity():
    """A small-looking total loss is still a high-severity claim."""
    score = score_severity(
        {**SIMPLE_CLAIM, "total_claim_amount": 5_000, "incident_severity": "Total Loss"}
    )
    assert score.band is Band.HIGH
    assert "SEV_TOTAL_LOSS" in score.reason_codes


def test_severity_always_explains_itself():
    score = score_severity(SIMPLE_CLAIM)
    assert score.reasons
    assert "SEV_AMOUNT" in score.reason_codes


# --- Complexity ---------------------------------------------------------------


def test_simple_claim_is_low_complexity():
    score = score_complexity(SIMPLE_CLAIM, CoverageStatus.CONFIRMED)
    assert score.band is Band.LOW
    assert score.value <= 3


def test_complex_claim_is_high_complexity():
    score = score_complexity(COMPLEX_CLAIM, CoverageStatus.CONFIRMED)
    assert score.band is Band.HIGH
    assert "CPX_INJURY" in score.reason_codes
    assert "CPX_MULTI_PARTY" in score.reason_codes


def test_coverage_problems_raise_complexity():
    confirmed = score_complexity(SIMPLE_CLAIM, CoverageStatus.CONFIRMED)
    question = score_complexity(SIMPLE_CLAIM, CoverageStatus.QUESTION)
    not_in_force = score_complexity(SIMPLE_CLAIM, CoverageStatus.NOT_IN_FORCE)
    assert question.value > confirmed.value
    assert not_in_force.value > question.value
    assert "CPX_COVERAGE_NOT_IN_FORCE" in not_in_force.reason_codes


def test_clean_claim_says_so_rather_than_returning_silence():
    score = score_complexity(SIMPLE_CLAIM, CoverageStatus.CONFIRMED)
    assert score.reasons
    assert "CPX_NONE" in score.reason_codes


def test_complexity_points_are_the_sum_of_its_reasons():
    score = score_complexity(COMPLEX_CLAIM, CoverageStatus.QUESTION)
    assert score.value == sum(r.points for r in score.reasons)


# --- Fraud --------------------------------------------------------------------


def test_fraud_bands_follow_the_probability():
    assert score_fraud(SIMPLE_CLAIM, 0.05).band is Band.LOW
    assert score_fraud(SIMPLE_CLAIM, 0.30).band is Band.MEDIUM
    assert score_fraud(SIMPLE_CLAIM, 0.80).band is Band.HIGH


def test_model_score_is_always_a_stated_reason():
    score = score_fraud(SIMPLE_CLAIM, 0.42)
    assert "FRD_MODEL_SCORE" in score.reason_codes
    assert "42%" in score.reason_texts[0]


def test_red_flags_fire_on_classic_indicators():
    claim = {
        **SIMPLE_CLAIM,
        "incident_hour_of_the_day": 3,
        "months_as_customer": 2,
        "missing_police_report": True,
        "witnesses": 0,
        "total_claim_amount": 80_000,
    }
    points, reasons = red_flags(claim)
    codes = {r.code for r in reasons}
    assert "FRD_LATE_NIGHT" in codes
    assert "FRD_SHORT_TENURE" in codes
    assert "FRD_NO_REPORT_HIGH_VALUE" in codes
    assert "FRD_NO_CORROBORATION" in codes
    assert points >= 3


def test_no_red_flags_on_a_clean_claim():
    points, reasons = red_flags(SIMPLE_CLAIM)
    assert points == 0
    assert reasons == []


def test_red_flags_can_escalate_the_band():
    """A low model score plus several red flags should not stay Low."""
    flagged = {
        **SIMPLE_CLAIM,
        "incident_hour_of_the_day": 2,
        "months_as_customer": 1,
        "missing_police_report": True,
        "witnesses": 0,
    }
    baseline = score_fraud(SIMPLE_CLAIM, 0.05)
    escalated = score_fraud(flagged, 0.05)
    assert baseline.band is Band.LOW
    assert escalated.band.rank > baseline.band.rank
    assert "FRD_FLAG_ESCALATION" in escalated.reason_codes


def test_red_flags_never_escalate_beyond_high():
    claim = {**SIMPLE_CLAIM, "incident_hour_of_the_day": 2, "months_as_customer": 1,
             "missing_police_report": True, "witnesses": 0}
    assert score_fraud(claim, 0.95).band is Band.HIGH


# --- Whole-claim triage -------------------------------------------------------


def test_triage_returns_all_three_scores():
    result = triage_claim(SIMPLE_CLAIM, 0.10, CoverageStatus.CONFIRMED)
    payload = result.to_dict()
    for key in ("severity_band", "complexity_band", "fraud_band", "coverage_status"):
        assert key in payload
    assert payload["triage_version"]


def test_every_score_carries_at_least_one_reason():
    """An unexplained score is unusable in claims."""
    result = triage_claim(COMPLEX_CLAIM, 0.7, CoverageStatus.QUESTION)
    for score in (result.severity, result.complexity, result.fraud):
        assert score.reasons
        assert all(r.text for r in score.reasons)


def test_triage_is_deterministic():
    a = triage_claim(COMPLEX_CLAIM, 0.7, CoverageStatus.QUESTION).to_dict()
    b = triage_claim(COMPLEX_CLAIM, 0.7, CoverageStatus.QUESTION).to_dict()
    assert a == b


def test_batch_triage_covers_every_claim(claims, policies):
    from claimpilot.coverage import check_coverage_batch
    from claimpilot.triage import triage_batch

    coverage = check_coverage_batch(claims, policies)
    result = triage_batch(claims, coverage)
    assert len(result) == len(claims)
    assert result["severity_band"].isin(["Low", "Medium", "High"]).all()
    assert result["complexity_band"].isin(["Low", "Medium", "High"]).all()
    assert result["fraud_band"].isin(["Low", "Medium", "High"]).all()
