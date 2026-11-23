"""Queue prioritisation, deadline tracking and the copilot briefing."""

from __future__ import annotations

from datetime import date

import pandas as pd

from claimpilot import config
from claimpilot.copilot import (
    ALLOWED_FIELDS,
    build_briefing,
    missing_documents,
    next_best_action,
)
from claimpilot.workbench import (
    DeadlineStatus,
    add_deadline_status,
    build_queue,
    deadline_status,
    demo_today,
    score_priority,
)

# --- Deadlines ----------------------------------------------------------------


def test_demo_date_is_fixed_and_inside_the_dataset_window(claims):
    today = demo_today()
    latest_loss = pd.to_datetime(claims["incident_date"]).max().date()
    assert today >= latest_loss, "demo date must be after the last loss"
    assert today.year == 2015, "a real current date would make every claim overdue"


def test_breached_at_risk_and_on_track():
    today = date(2015, 3, 15)
    assert deadline_status(date(2015, 3, 1), today) is DeadlineStatus.BREACHED
    assert deadline_status(date(2015, 6, 1), today) is DeadlineStatus.ON_TRACK
    # Within the final 20% of a 30-day window.
    assert deadline_status(date(2015, 3, 18), today, 30) is DeadlineStatus.AT_RISK


def test_due_today_is_not_yet_breached():
    today = date(2015, 3, 15)
    assert deadline_status(today, today) is not DeadlineStatus.BREACHED


def test_met_obligations_are_not_counted_as_breaches(deadlines):
    """Digital FNOL acknowledges on submission, so that clock is satisfied."""
    frame = add_deadline_status(deadlines)
    assert (frame["acknowledge_status"] == "Met").all()


def test_headline_status_follows_the_next_open_obligation(deadlines):
    """Regression: taking the worst of all three marked all 1,000 breached."""
    frame = add_deadline_status(deadlines)
    assert frame["deadline_status"].nunique() > 1, (
        "every claim has the same deadline status; the tracker says nothing"
    )
    assert set(frame["deadline_status"]) <= {s.value for s in DeadlineStatus}


def test_deadline_mix_is_informative(deadlines):
    frame = add_deadline_status(deadlines)
    share_breached = (frame["deadline_status"] == "Breached").mean()
    assert share_breached < 1.0
    assert (frame["deadline_status"] == "On track").any()


def test_age_and_days_to_due_are_computed(deadlines):
    frame = add_deadline_status(deadlines)
    assert (frame["age_days"] >= 0).all()
    assert frame["days_to_decide_due"].notna().all()


# --- Priority -----------------------------------------------------------------


def test_priority_rewards_urgency():
    low = score_priority(
        {"severity_band": "Low", "deadline_status": "On track",
         "fraud_band": "Low", "complexity_band": "Low",
         "coverage_status": "Confirmed", "age_days": 1}
    )
    high = score_priority(
        {"severity_band": "High", "deadline_status": "Breached",
         "fraud_band": "High", "complexity_band": "High",
         "coverage_status": "Coverage question", "age_days": 60}
    )
    assert high.value > low.value


def test_priority_explains_itself():
    score = score_priority(
        {"severity_band": "High", "deadline_status": "Breached",
         "fraud_band": "Low", "complexity_band": "Low",
         "coverage_status": "Confirmed", "age_days": 10}
    )
    assert score.reasons
    assert any("severity" in r.lower() for r in score.reasons)
    assert any("deadline" in r.lower() for r in score.reasons)


def test_priority_weights_come_from_config():
    weights = config.assumption("workbench", "priority_weights")
    score = score_priority(
        {"severity_band": "High", "deadline_status": "On track",
         "fraud_band": "Low", "complexity_band": "Low",
         "coverage_status": "Confirmed", "age_days": 0}
    )
    assert score.value == float(weights["severity_high"])


def test_queue_is_sorted_by_priority(scored_portfolio, claims, deadlines):
    queue = build_queue(scored_portfolio, claims, deadlines)
    assert (queue["priority_score"].diff().dropna() <= 0).all()


def test_queue_can_be_filtered_by_destination(scored_portfolio, claims, deadlines):
    siu = build_queue(scored_portfolio, claims, deadlines, ["siu_queue"])
    assert len(siu) > 0
    assert (siu["destination"] == "siu_queue").all()


def test_queue_excludes_nothing_by_default(scored_portfolio, claims, deadlines):
    queue = build_queue(scored_portfolio, claims, deadlines)
    assert len(queue) == len(scored_portfolio)


# --- Copilot ------------------------------------------------------------------


def test_copilot_sees_only_allowed_fields(claims):
    """It must not be able to read the policyholder's demographics."""
    for field in ("insured_sex", "age", "insured_zip", "insured_occupation",
                  "insured_education_level", "insured_hobbies"):
        assert field not in ALLOWED_FIELDS


def test_missing_documents_follow_the_claim_record():
    docs = missing_documents(
        {"missing_police_report": True, "has_injury": True, "bodily_injuries": 2,
         "coverage_status": "Confirmed", "total_claim_amount": 50_000,
         "severity_band": "Medium"}
    )
    names = {d["document"] for d in docs}
    assert "Police report" in names
    assert "Medical records" in names
    assert all(d["why"] for d in docs)


def test_clean_claim_has_no_outstanding_documents():
    docs = missing_documents(
        {"missing_police_report": False, "missing_property_damage_info": False,
         "has_injury": False, "coverage_status": "Confirmed",
         "total_claim_amount": 3_000, "severity_band": "Low"}
    )
    assert docs == []


def test_next_action_prioritises_coverage_then_fraud():
    coverage = next_best_action(
        {"coverage_status": "Not in force at date of loss", "fraud_band": "High"}, []
    )
    assert "coverage" in coverage.lower() or "policy term" in coverage.lower()
    assert "cannot deny" in coverage.lower()

    fraud = next_best_action({"coverage_status": "Confirmed", "fraud_band": "High"}, [])
    assert "siu" in fraud.lower()


def test_briefing_is_assembled_from_the_claim(claims):
    briefing = build_briefing(claims.iloc[0])
    assert briefing.claim_id
    assert briefing.summary
    assert briefing.next_best_action
    assert briefing.reserve.point_usd >= 0


def test_briefing_summary_only_cites_amounts_from_the_record(claims):
    """The copilot must not invent a figure."""
    claim = claims.iloc[0]
    briefing = build_briefing(claim)
    actual = f"{claim['total_claim_amount']:,.0f}"
    assert actual in briefing.summary


def test_briefing_is_deterministic(claims):
    a = build_briefing(claims.iloc[2])
    b = build_briefing(claims.iloc[2])
    assert a.summary == b.summary
    assert a.reserve.point_usd == b.reserve.point_usd


def test_briefing_masks_pii_in_the_narrative(claims, narratives):
    claim = claims.iloc[0]
    narrative = (
        "My name is Jane Doe, jane.doe@example.com, (555) 213-4455. "
        + narratives.iloc[0]["narrative"]
    )
    briefing = build_briefing(claim, narrative=narrative)
    assert briefing.pii_items_masked >= 3
    assert "jane.doe@example.com" not in briefing.summary
