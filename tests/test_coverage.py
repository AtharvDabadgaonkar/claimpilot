"""Coverage checking against the mock policy."""

from __future__ import annotations

from datetime import date

import pytest

from claimpilot.coverage import (
    CoverageStatus,
    check_coverage,
    check_coverage_batch,
)

FULL_COVER = {
    "coverage_type": "Comprehensive + Collision + Liability",
    "term_start": date(2014, 6, 1),
    "term_end": date(2015, 6, 1),
    "deductible_usd": 1000,
    "csl_per_person_usd": 250_000,
    "csl_per_occurrence_usd": 500_000,
}
LIABILITY_ONLY = {**FULL_COVER, "coverage_type": "Liability Only"}


def test_in_force_claim_is_confirmed():
    result = check_coverage(FULL_COVER, date(2015, 1, 15), "Single Vehicle Collision", 20_000)
    assert result.status is CoverageStatus.CONFIRMED
    assert result.is_confirmed
    assert not result.blocks_fast_track
    assert result.deductible_usd == 1000


def test_expired_policy_is_flagged():
    """Required: a policy with expired coverage is flagged."""
    result = check_coverage(FULL_COVER, date(2015, 8, 1), "Single Vehicle Collision", 20_000)
    assert result.status is CoverageStatus.NOT_IN_FORCE
    assert result.blocks_fast_track
    assert result.days_outside_term == 61
    assert "expired" in " ".join(result.reasons).lower()


def test_loss_before_inception_is_flagged():
    result = check_coverage(FULL_COVER, date(2014, 1, 1), "Single Vehicle Collision", 5_000)
    assert result.status is CoverageStatus.NOT_IN_FORCE
    assert result.days_outside_term == 151
    assert "predates" in " ".join(result.reasons).lower()


def test_loss_on_the_first_day_of_cover_is_in_force():
    """Boundary: inception day itself is covered."""
    result = check_coverage(FULL_COVER, date(2014, 6, 1), "Single Vehicle Collision", 5_000)
    assert result.status is CoverageStatus.CONFIRMED


def test_loss_on_the_last_day_of_cover_is_in_force():
    result = check_coverage(FULL_COVER, date(2015, 6, 1), "Single Vehicle Collision", 5_000)
    assert result.status is CoverageStatus.CONFIRMED


@pytest.mark.parametrize(
    "loss_type",
    ["Single Vehicle Collision", "Multi-vehicle Collision", "Parked Car"],
)
def test_liability_only_raises_a_question_for_own_damage(loss_type):
    result = check_coverage(LIABILITY_ONLY, date(2015, 1, 15), loss_type, 20_000)
    assert result.status is CoverageStatus.QUESTION
    assert result.blocks_fast_track
    assert "liability" in " ".join(result.reasons).lower()


def test_liability_only_raises_a_question_for_theft():
    result = check_coverage(LIABILITY_ONLY, date(2015, 1, 15), "Vehicle Theft", 20_000)
    assert result.status is CoverageStatus.QUESTION
    assert "comprehensive" in " ".join(result.reasons).lower()


def test_claim_over_the_limit_raises_a_question():
    result = check_coverage(
        FULL_COVER, date(2015, 1, 15), "Single Vehicle Collision", 600_000
    )
    assert result.status is CoverageStatus.QUESTION
    assert "exceeds" in " ".join(result.reasons).lower()


def test_missing_dates_yield_unknown_not_a_denial():
    result = check_coverage(
        {**FULL_COVER, "term_end": None}, date(2015, 1, 15), "Vehicle Theft", 1_000
    )
    assert result.status is CoverageStatus.UNKNOWN
    assert result.blocks_fast_track


def test_every_result_carries_reasons():
    for policy, loss_date in [
        (FULL_COVER, date(2015, 1, 15)),
        (FULL_COVER, date(2015, 8, 1)),
        (LIABILITY_ONLY, date(2015, 1, 15)),
    ]:
        result = check_coverage(policy, loss_date, "Single Vehicle Collision", 20_000)
        assert result.reasons, "a coverage result must always explain itself"


def test_coverage_never_returns_a_denial():
    """Coverage raises questions; it never decides the claim."""
    statuses = {s.value for s in CoverageStatus}
    assert not any(
        word in s.lower() for s in statuses for word in ("deny", "denied", "reject")
    )


def test_batch_covers_every_claim(claims, policies):
    result = check_coverage_batch(claims, policies)
    assert len(result) == len(claims)
    assert set(result["claim_id"]) == set(claims["claim_id"])
    assert result["coverage_status"].notna().all()


def test_batch_flags_the_expired_policies_we_planted(claims, policies):
    """The simulated expired policies must actually be caught."""
    result = check_coverage_batch(claims, policies)
    expired_ids = set(policies.loc[policies["expected_expired_at_loss"], "claim_id"])
    flagged = result.loc[
        result["coverage_status"] == CoverageStatus.NOT_IN_FORCE.value, "claim_id"
    ]
    assert expired_ids == set(flagged)
