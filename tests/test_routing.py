"""Routing: rule matching, and the hard no-auto-denial guarantee."""

from __future__ import annotations

import pytest
import yaml

from claimpilot import config
from claimpilot.routing import (
    AutoDenialError,
    Destination,
    assert_no_auto_denial_path,
    load_rules,
    route,
    route_batch,
)


def triage_row(**overrides):
    base = {
        "claim_id": "TEST-001",
        "severity_band": "Low",
        "complexity_band": "Low",
        "fraud_band": "Low",
        "coverage_status": "Confirmed",
    }
    base.update(overrides)
    return base


# =============================================================================
# THE HARD RULE: auto-approve yes, auto-deny never.
# =============================================================================


def test_no_denial_destination_exists_in_the_enum():
    """A denial cannot even be represented as a routing outcome."""
    values = {d.value.lower() for d in Destination}
    for word in ("deny", "denied", "denial", "reject", "rejected", "close_without"):
        assert not any(word in v for v in values), f"'{word}' is representable"


def test_no_auto_denial_path_guarantee():
    """Required: there is no code path producing an automatic denial."""
    assert_no_auto_denial_path()


def test_no_rule_routes_to_a_forbidden_destination():
    rules = load_rules()
    forbidden = {d.lower() for d in rules["forbidden_destinations"]}
    reachable = {r["destination"].lower() for r in rules["rules"]}
    reachable.add(rules["fallback"]["destination"].lower())
    assert not (reachable & forbidden)


def test_every_routing_outcome_over_the_whole_portfolio_is_non_denial(claims, policies):
    """Exhaustive check: route all 1,000 claims, assert none is a denial."""
    from claimpilot.coverage import check_coverage_batch
    from claimpilot.triage import triage_batch

    coverage = check_coverage_batch(claims, policies)
    routed = route_batch(triage_batch(claims, coverage))
    assert set(routed["destination"]) <= {d.value for d in Destination}
    assert routed["destination"].notna().all()
    # Only STP is automated; everything else reaches a human.
    automated = routed.loc[routed["auto_approved"], "destination"].unique()
    assert set(automated) <= {Destination.STP_AUTO_APPROVE.value}


def test_the_only_automated_outcome_is_an_approval():
    for destination in Destination:
        if destination.is_automated:
            assert destination is Destination.STP_AUTO_APPROVE


def test_loader_rejects_a_rules_file_containing_a_denial(tmp_path, monkeypatch):
    """If someone adds a denial destination later, the engine refuses to load."""
    rules = yaml.safe_load(config.ROUTING_RULES_PATH.read_text())
    rules["rules"].insert(
        0,
        {
            "name": "Sneaky auto-deny",
            "when": {"coverage_status": ["Not in force at date of loss"]},
            "destination": "auto_deny",
            "reason": "should never be allowed",
        },
    )
    bad = tmp_path / "routing_rules.yaml"
    bad.write_text(yaml.safe_dump(rules))

    monkeypatch.setattr(config, "ROUTING_RULES_PATH", bad)
    load_rules.cache_clear()
    try:
        with pytest.raises(AutoDenialError, match="never auto-deny"):
            load_rules()
    finally:
        load_rules.cache_clear()


def test_loader_rejects_an_undeclared_destination(tmp_path, monkeypatch):
    rules = yaml.safe_load(config.ROUTING_RULES_PATH.read_text())
    rules["fallback"]["destination"] = "somewhere_else"
    bad = tmp_path / "routing_rules.yaml"
    bad.write_text(yaml.safe_dump(rules))

    monkeypatch.setattr(config, "ROUTING_RULES_PATH", bad)
    load_rules.cache_clear()
    try:
        with pytest.raises(AutoDenialError, match="undeclared destination"):
            load_rules()
    finally:
        load_rules.cache_clear()


# =============================================================================
# Rule matching on hand-built claims
# =============================================================================


def test_simple_covered_claim_is_fast_tracked():
    decision = route(triage_row())
    assert decision.destination is Destination.STP_AUTO_APPROVE
    assert decision.is_auto_approved
    assert decision.rule_name == "STP fast-track"


def test_high_fraud_goes_to_siu():
    decision = route(triage_row(fraud_band="High"))
    assert decision.destination is Destination.SIU_QUEUE
    assert decision.destination.requires_human


def test_high_complexity_goes_to_a_senior_adjuster():
    decision = route(triage_row(complexity_band="High"))
    assert decision.destination is Destination.SENIOR_ADJUSTER


def test_medium_claim_goes_to_the_adjuster_queue():
    decision = route(triage_row(severity_band="Medium", complexity_band="Medium"))
    assert decision.destination is Destination.ADJUSTER_QUEUE
    assert decision.rule_name == "fallback"


def test_coverage_not_in_force_goes_to_a_human_not_a_denial():
    """The most likely denial path must still reach a person."""
    decision = route(triage_row(coverage_status="Not in force at date of loss"))
    assert decision.destination is Destination.SENIOR_ADJUSTER
    assert decision.destination.requires_human
    assert not decision.is_auto_approved
    assert "cannot deny" in decision.reason.lower()


def test_coverage_question_blocks_the_fast_track():
    decision = route(triage_row(coverage_status="Coverage question"))
    assert decision.destination is not Destination.STP_AUTO_APPROVE


def test_unknown_coverage_blocks_the_fast_track():
    decision = route(triage_row(coverage_status="Unknown"))
    assert decision.destination is not Destination.STP_AUTO_APPROVE


@pytest.mark.parametrize(
    "overrides",
    [
        {"severity_band": "Medium"},
        {"complexity_band": "Medium"},
        {"fraud_band": "Medium"},
        {"coverage_status": "Coverage question"},
        {"coverage_status": "Not in force at date of loss"},
    ],
)
def test_fast_track_requires_every_condition(overrides):
    """STP needs all four green lights; any one failing stops it."""
    decision = route(triage_row(**overrides))
    assert decision.destination is not Destination.STP_AUTO_APPROVE


def test_fraud_is_checked_before_complexity():
    """A high-fraud, high-complexity claim belongs to SIU, not a senior adjuster."""
    decision = route(triage_row(fraud_band="High", complexity_band="High"))
    assert decision.destination is Destination.SIU_QUEUE


def test_coverage_not_in_force_outranks_high_fraud():
    """Order matters: the coverage question is the first thing a human must see."""
    decision = route(
        triage_row(coverage_status="Not in force at date of loss", fraud_band="High")
    )
    assert decision.destination is Destination.SENIOR_ADJUSTER


def test_every_decision_explains_itself():
    for overrides in ({}, {"fraud_band": "High"}, {"severity_band": "Medium"}):
        decision = route(triage_row(**overrides))
        assert decision.reason
        assert decision.rule_name
        assert decision.rules_version


def test_high_severity_simple_claim_still_gets_a_senior_adjuster():
    decision = route(triage_row(severity_band="High"))
    assert decision.destination is Destination.SENIOR_ADJUSTER
