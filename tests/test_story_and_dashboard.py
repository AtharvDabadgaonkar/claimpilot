"""The Overview story beats and the KPI dashboard view.

The dashboard is deliberately a *view*: every tile is read from a section of
metrics.json rather than recomputed. The point of these tests is that a tile
cannot quietly disagree with the section it summarises.
"""

from __future__ import annotations

import pytest

from claimpilot import config
from claimpilot.kpis import load_metrics, ops_dashboard_metrics
from claimpilot.routing import Destination
from claimpilot.story import pick_story_claims, story_frame

# ---------------------------------------------------------------------------
# Story beats
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def beats(scored_portfolio, claims):
    return pick_story_claims(scored_portfolio, claims)


def test_three_beats_one_per_outcome(beats):
    assert len(beats) == 3
    dests = [b.destination for b in beats]
    assert dests[0] == Destination.STP_AUTO_APPROVE.value
    assert dests[1] == Destination.SIU_QUEUE.value
    assert dests[2] in (
        Destination.ADJUSTER_QUEUE.value,
        Destination.SENIOR_ADJUSTER.value,
    )


def test_the_fast_tracked_beat_needs_no_human(beats):
    first = beats[0]
    assert first.auto_approved is True
    assert first.requires_human is False
    assert first.severity_band == "Low"
    assert first.fraud_band == "Low"


def test_the_siu_beat_goes_to_a_person_not_a_denial(beats):
    """Even the most suspicious claim in the book is referred, never denied."""
    siu = beats[1]
    assert siu.requires_human is True
    assert siu.auto_approved is False
    assert siu.fraud_band == "High"
    assert "refer" in siu.routing_reason.lower()


def test_no_beat_is_a_denial(beats):
    allowed = {d.value for d in Destination}
    for b in beats:
        assert b.destination in allowed
        assert "deny" not in b.destination
        assert "deny" not in b.routing_reason.lower()


def test_beats_are_picked_deterministically(scored_portfolio, claims):
    first = pick_story_claims(scored_portfolio, claims)
    second = pick_story_claims(scored_portfolio, claims)
    assert [b.claim_id for b in first] == [b.claim_id for b in second]


def test_beats_are_picked_by_rule_not_by_id(scored_portfolio, claims):
    """Shuffling the input must not change which claims the story picks."""
    shuffled = scored_portfolio.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert [b.claim_id for b in pick_story_claims(shuffled, claims)] == [
        b.claim_id for b in pick_story_claims(scored_portfolio, claims)
    ]


def test_the_siu_beat_is_the_highest_fraud_signal_in_the_book(beats, scored_portfolio):
    siu = scored_portfolio[scored_portfolio["destination"] == Destination.SIU_QUEUE.value]
    assert beats[1].fraud_probability == pytest.approx(
        siu["fraud_probability"].max(), abs=1e-9
    )


def test_story_survives_an_empty_portfolio(scored_portfolio, claims):
    """A page must not crash if a route happens to be empty."""
    empty = scored_portfolio.iloc[0:0]
    assert pick_story_claims(empty, claims) == []


def test_story_frame_is_displayable(beats):
    frame = story_frame(beats)
    assert len(frame) == 3
    for col in ("Beat", "Claim", "Routed to", "Rule fired", "Human decides"):
        assert col in frame.columns
    assert set(frame["Human decides"]) <= {"Yes", "No"}


# ---------------------------------------------------------------------------
# Dashboard tiles must not drift from their sources
# ---------------------------------------------------------------------------


def test_kpi_tiles_match_their_source_sections():
    m = load_metrics()
    k = m["ops_dashboard"]["kpis"]
    assert k["stp_rate"] == m["triage_routing"]["stp_rate"]
    assert k["reserve_mae_usd"] == m["reserve_model"]["mae_usd"]
    assert k["reserve_interval_coverage"] == m["reserve_model"]["interval_coverage"]
    assert k["siu_hit_rate"] == m["backtest"]["siu"]["hit_rate_precision"]
    assert k["lae_per_claim_usd"] == m["backtest"]["arms"]["claimpilot"]["mean_lae_usd"]
    assert (
        k["mean_cycle_days"] == m["backtest"]["arms"]["claimpilot"]["mean_cycle_days"]
    )


def test_guardrail_tiles_match_their_source_sections():
    m = load_metrics()
    g = m["ops_dashboard"]["guardrails"]
    gov = m["governance_programme"]
    assert g["override_rate"] == m["workbench"]["override_rate"]
    assert g["deadline_breached_rate"] == m["workbench"]["deadline_breached_rate"]
    assert g["worst_fairness_ratio"] == gov["fairness"]["worst_ratio"]
    assert g["fairness_tests_flagged"] == gov["fairness"]["tests_flagged"]
    assert g["monitoring_alerts"] == gov["monitoring"]["alerts"]
    assert (
        g["estimated_leakage_total_usd"]
        == m["backtest"]["arms"]["claimpilot"]["total_leakage_usd"]
    )


def test_dashboard_view_recomputes_nothing_it_publishes():
    """Build the view twice; identical output means it is a pure read."""
    assert ops_dashboard_metrics() == ops_dashboard_metrics()


def test_auto_denied_count_is_zero_on_every_surface():
    m = load_metrics()
    assert m["ops_dashboard"]["guardrails"]["auto_denied_count"] == 0
    assert m["triage_routing"]["auto_denied_count"] == 0
    assert m["governance"]["auto_deny_paths"] == 0


def test_guardrails_are_published_beside_the_kpis():
    """A speed number with no guardrail next to it is the failure mode."""
    ops = load_metrics()["ops_dashboard"]
    assert set(ops) >= {"kpis", "guardrails", "guardrail_note"}
    for required in (
        "estimated_leakage_total_usd",
        "worst_fairness_ratio",
        "override_rate",
        "deadline_breached_rate",
        "auto_denied_count",
        "reopen_rate",
    ):
        assert required in ops["guardrails"]


def test_the_failing_guardrails_are_still_reported_as_failing():
    """A regression guard, not a target.

    Two guardrails currently breach their own thresholds. If a future change
    makes them pass, that must be because the behaviour improved — not because
    a threshold moved or a metric quietly stopped being published.
    """
    g = load_metrics()["ops_dashboard"]["guardrails"]
    assert g["fairness_tests_flagged"] > 0, (
        "fairness now passes everywhere; confirm the check still runs and the "
        "0.80 threshold was not loosened before updating this test"
    )
    assert g["worst_fairness_ratio"] < float(
        config.assumption("fairness", "disparate_impact_threshold")
    )
    assert g["deadline_breached_rate"] > 0


def test_metrics_record_phase_five_as_complete():
    m = load_metrics()
    assert "5-dashboard-and-ship" in m["phases_complete"]
    assert "backtest" in m
    assert "ops_dashboard" in m
