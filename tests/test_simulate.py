"""Simulated layers must be reproducible, labelled, and internally consistent."""

from __future__ import annotations

import pandas as pd

from claimpilot import config, simulate


def test_every_simulated_table_is_labelled(policies, narratives, deadlines, baseline):
    """A reviewer must be able to tell simulated rows apart at a glance."""
    for name, df in (
        ("policies", policies),
        ("narratives", narratives),
        ("deadlines", deadlines),
        ("baseline", baseline),
    ):
        assert "is_simulated" in df.columns, f"{name} missing is_simulated"
        assert df["is_simulated"].all()


def test_one_row_per_claim(claims, policies, narratives, deadlines, baseline):
    ids = set(claims["claim_id"])
    for df in (policies, narratives, deadlines, baseline):
        assert set(df["claim_id"]) == ids
        assert df["claim_id"].is_unique


def test_simulation_is_deterministic(claims):
    """Same seed, same output — the committed Parquet must be reproducible."""
    a = simulate.simulate_contacts(claims)
    b = simulate.simulate_contacts(claims)
    pd.testing.assert_frame_equal(a, b)

    pd.testing.assert_frame_equal(
        simulate.simulate_policies(claims), simulate.simulate_policies(claims)
    )
    pd.testing.assert_frame_equal(
        simulate.simulate_manual_baseline(claims),
        simulate.simulate_manual_baseline(claims),
    )


def test_contacts_use_reserved_domains_and_area_codes(policies):
    """Fictional contact details must never be deliverable."""
    assert policies["policyholder_email"].str.endswith("@example.com").all()
    assert policies["policyholder_phone"].str.startswith("(555)").all()


def test_mock_policy_numbers_are_prefixed(policies):
    assert policies["mock_policy_number"].str.startswith("NWM-").all()


def test_some_policies_are_expired_at_loss(policies, claims):
    """The coverage check needs real failures, not an always-pass demo."""
    expired = policies[policies["expected_expired_at_loss"]]
    assert len(expired) > 0, "no expired policies generated"

    merged = expired.merge(claims[["claim_id", "incident_date"]], on="claim_id")
    # Term really does end before the date of loss.
    assert (
        pd.to_datetime(merged["term_end"]) < pd.to_datetime(merged["incident_date"])
    ).all()


def test_in_force_policies_actually_cover_the_loss_date(policies, claims):
    in_force = policies[~policies["expected_expired_at_loss"]]
    merged = in_force.merge(claims[["claim_id", "incident_date"]], on="claim_id")
    loss = pd.to_datetime(merged["incident_date"])
    assert (pd.to_datetime(merged["term_start"]) <= loss).all()
    assert (pd.to_datetime(merged["term_end"]) >= loss).all()


def test_some_policies_are_liability_only(policies):
    """These create genuine coverage questions for physical damage claims."""
    assert policies["expected_liability_only"].sum() > 0


def test_narratives_are_substantive(narratives):
    assert (narratives["char_count"] > 120).all()
    assert narratives["narrative"].str.contains("It happened on").all()


def test_narrative_ground_truth_matches_source_claim(claims, narratives):
    """Ground truth is what makes extraction accuracy measurable, not asserted."""
    merged = narratives.merge(claims, on="claim_id")
    assert (merged["truth_loss_type"] == merged["incident_type"]).all()
    assert (merged["truth_num_vehicles"] == merged["number_of_vehicles_involved"]).all()
    assert (merged["truth_auto_make"] == merged["auto_make"]).all()
    assert (merged["truth_num_injured"] == merged["bodily_injuries"]).all()
    assert (
        merged["truth_injury_reported"] == (merged["bodily_injuries"] > 0)
    ).all()
    assert (
        pd.to_datetime(merged["truth_loss_date"])
        == pd.to_datetime(merged["incident_date"])
    ).all()


def test_narrative_states_the_real_amount_when_not_vague(claims, narratives):
    merged = narratives.merge(claims, on="claim_id")
    stated = merged[merged["truth_stated_amount_usd"].notna()]
    assert len(stated) > 0
    assert (stated["truth_stated_amount_usd"] == stated["total_claim_amount"]).all()


def test_some_narratives_omit_the_amount(narratives):
    """The extractor must cope with vague claimants."""
    vague = narratives["truth_stated_amount_usd"].isna()
    assert vague.sum() > 0
    assert vague.mean() < 0.5


def test_pii_appears_in_a_minority_of_narratives(narratives):
    rate = narratives["truth_contains_pii"].mean()
    target = float(config.assumption("narratives", "include_pii_rate"))
    assert abs(rate - target) < 0.05


def test_deadline_ordering(deadlines):
    ack = pd.to_datetime(deadlines["acknowledge_due"])
    decide = pd.to_datetime(deadlines["decide_due"])
    pay = pd.to_datetime(deadlines["pay_due"])
    reported = pd.to_datetime(deadlines["reported_date"])
    assert (reported < ack).all()
    assert (ack < decide).all()
    assert (decide < pay).all()


def test_deadlines_declare_they_are_illustrative(deadlines):
    assert deadlines["deadline_basis"].str.contains("ILLUSTRATIVE").all()


def test_baseline_values_are_positive_and_plausible(baseline):
    assert (baseline["baseline_handling_hours"] > 0).all()
    assert (baseline["baseline_cycle_days"] > 0).all()
    assert (baseline["baseline_lae_usd"] > 0).all()
    # LAE must reflect adjuster hours plus the fixed per-claim cost.
    rate = float(config.assumption("costs", "adjuster_cost_per_hour_usd"))
    fixed = float(config.assumption("costs", "fixed_cost_per_claim_usd"))
    expected = baseline["baseline_handling_hours"] * rate + fixed
    assert ((baseline["baseline_lae_usd"] - expected).abs() < 1.0).all()


def test_baseline_severity_bands_ordered(baseline):
    """More severe claims must cost more to handle, on average."""
    means = baseline.groupby("severity_band")["baseline_handling_hours"].mean()
    assert means["low_severity"] < means["high_severity"]
