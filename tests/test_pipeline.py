"""End-to-end scoring pipeline over the whole portfolio."""

from __future__ import annotations

import pytest

from claimpilot.pipeline import SCORED_PARQUET, load_scored, score_portfolio
from claimpilot.routing import Destination


@pytest.fixture(scope="module")
def scored():
    return load_scored()


def test_scored_file_is_committed_and_small():
    assert SCORED_PARQUET.exists(), "run `make score`"
    assert SCORED_PARQUET.stat().st_size / 1024 / 1024 < 25


def test_every_claim_is_scored_and_routed(scored, claims):
    assert len(scored) == len(claims)
    assert set(scored["claim_id"]) == set(claims["claim_id"])
    assert scored["destination"].notna().all()


def test_committed_scores_match_a_fresh_run(scored):
    """The committed artifact must be reproducible from the committed inputs."""
    fresh = score_portfolio()
    assert list(fresh["claim_id"]) == list(scored["claim_id"])
    for col in ("severity_band", "complexity_band", "fraud_band",
                "coverage_status", "destination"):
        assert (fresh[col].to_numpy() == scored[col].to_numpy()).all(), f"{col} drifted"


def test_no_claim_is_auto_denied(scored):
    assert set(scored["destination"]) <= {d.value for d in Destination}
    assert scored.loc[scored["auto_approved"], "destination"].eq(
        Destination.STP_AUTO_APPROVE.value
    ).all()


def test_auto_approved_claims_all_had_coverage_confirmed(scored):
    """Nothing is fast-tracked with an open coverage question."""
    approved = scored[scored["auto_approved"]]
    assert (approved["coverage_status"] == "Confirmed").all()


def test_auto_approved_claims_are_all_low_risk(scored):
    approved = scored[scored["auto_approved"]]
    assert (approved["severity_band"] == "Low").all()
    assert (approved["complexity_band"] == "Low").all()
    assert (approved["fraud_band"] == "Low").all()


def test_every_scored_claim_carries_reasons(scored):
    for col in ("severity_reasons", "complexity_reasons", "fraud_reasons",
                "coverage_reasons"):
        assert scored[col].apply(len).gt(0).all(), f"{col} empty on some claims"


def test_routing_reason_present_on_every_claim(scored):
    assert scored["routing_reason"].str.len().gt(0).all()


def test_versions_are_recorded_for_audit(scored):
    assert scored["triage_version"].nunique() == 1
    assert scored["rules_version"].nunique() == 1
