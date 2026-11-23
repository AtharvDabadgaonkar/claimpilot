"""The backtest, the demo story and the KPI dashboard view.

The backtest is assumption-driven by construction, so these tests focus on the
things that *can* be wrong: the pairing, the arithmetic, the leakage logic, and
whether the dashboard tiles agree with the sections they claim to summarise.
"""

from __future__ import annotations

import pandas as pd
import pytest

from claimpilot import config
from claimpilot.backtest import (
    BACKTEST_VERSION,
    build_arms,
    by_route,
    comparison_frame,
    deltas,
    lae_breakeven,
    siu_hit_rate,
    summary,
)

# ---------------------------------------------------------------------------
# Pairing: the ClaimPilot arm must come from the baseline draw, not a new one
# ---------------------------------------------------------------------------


def test_every_baseline_claim_appears_exactly_once(backtest_arms, baseline):
    assert len(backtest_arms) == len(baseline)
    assert backtest_arms["claim_id"].is_unique
    assert set(backtest_arms["claim_id"]) == set(baseline["claim_id"])


def test_build_arms_is_deterministic(baseline, scored_portfolio, claims):
    """No random draw anywhere in the ClaimPilot arm."""
    first = build_arms(baseline, scored_portfolio, claims)
    second = build_arms(baseline, scored_portfolio, claims)
    pd.testing.assert_frame_equal(first, second)


def test_build_arms_raises_when_rows_are_lost(baseline, scored_portfolio, claims):
    """A silent inner-join drop would corrupt every total. It must fail loudly."""
    with pytest.raises(ValueError, match="lost rows"):
        build_arms(baseline, scored_portfolio.iloc[:500], claims)


def test_assisted_claims_scale_their_own_baseline_draw(backtest_arms):
    """An assisted claim's hours are its own draw times the multiplier."""
    mult = float(
        config.assumption(
            "handling_time_hours", "claimpilot", "adjuster_assisted_multiplier"
        )
    )
    assisted = backtest_arms[
        ~backtest_arms["fast_tracked"] & ~backtest_arms["routed_to_siu"]
    ]
    assert len(assisted) > 100
    expected = assisted["baseline_handling_hours"] * mult
    # 0.01 tolerance: the stored column is rounded to 2dp.
    assert (expected - assisted["claimpilot_handling_hours"]).abs().max() < 0.011


# ---------------------------------------------------------------------------
# Route-specific behaviour
# ---------------------------------------------------------------------------


def test_fast_tracked_claims_collapse_to_the_assumed_touch_time(backtest_arms):
    stp_hours = float(
        config.assumption("handling_time_hours", "claimpilot", "stp_auto_approved")
    )
    fast = backtest_arms[backtest_arms["fast_tracked"]]
    assert len(fast) > 0
    assert (fast["claimpilot_adjuster_hours"] - stp_hours).abs().max() < 1e-6
    assert (fast["claimpilot_siu_hours"] == 0).all()


def test_siu_time_is_charged_only_to_referred_claims(backtest_arms):
    extra = float(
        config.assumption("handling_time_hours", "claimpilot", "siu_referral_extra")
    )
    referred = backtest_arms[backtest_arms["routed_to_siu"]]
    not_referred = backtest_arms[~backtest_arms["routed_to_siu"]]
    assert len(referred) > 0
    assert (referred["claimpilot_siu_hours"] - extra).abs().max() < 1e-6
    assert (not_referred["claimpilot_siu_hours"] == 0).all()


def test_siu_referrals_cost_more_than_the_manual_arm(backtest_arms):
    """The uncomfortable half of the result, asserted so it cannot be lost.

    The all-manual arm investigates nothing, so a referral is pure added
    expense. If a future change makes referrals look cheap, something has
    quietly stopped charging for investigation.
    """
    referred = backtest_arms[backtest_arms["routed_to_siu"]]
    assert referred["claimpilot_lae_usd"].mean() > referred["baseline_lae_usd"].mean()


def test_fast_tracked_claims_are_cheaper_and_faster(backtest_arms):
    fast = backtest_arms[backtest_arms["fast_tracked"]]
    assert fast["claimpilot_lae_usd"].mean() < fast["baseline_lae_usd"].mean()
    assert fast["claimpilot_cycle_days"].mean() < fast["baseline_cycle_days"].mean()


def test_fast_tracked_lae_uses_the_automated_system_cost(backtest_arms):
    costs = config.assumption("costs")
    expected = (
        float(
            config.assumption("handling_time_hours", "claimpilot", "stp_auto_approved")
        )
        * float(costs["adjuster_cost_per_hour_usd"])
        + float(costs["stp_system_cost_per_claim_usd"])
    )
    fast = backtest_arms[backtest_arms["fast_tracked"]]
    assert fast["claimpilot_lae_usd"].max() == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


def test_unflagged_claims_leak_nothing_unless_fast_tracked(backtest_arms):
    quiet = backtest_arms[
        ~backtest_arms["fraud_reported"].astype(bool)
        & ~backtest_arms["fast_tracked"]
    ]
    assert len(quiet) > 100
    assert (quiet["baseline_leakage_usd"] == 0).all()
    assert (quiet["claimpilot_leakage_usd"] == 0).all()


def test_referring_a_flagged_claim_reduces_its_leakage(backtest_arms):
    flagged = backtest_arms[backtest_arms["fraud_reported"].astype(bool)]
    referred = flagged[flagged["routed_to_siu"]]
    missed = flagged[~flagged["routed_to_siu"] & ~flagged["fast_tracked"]]
    assert len(referred) > 0 and len(missed) > 0
    # Compare as a share of the claim, since amounts differ between groups.
    ref_share = (
        referred["claimpilot_leakage_usd"] / referred["total_claim_amount"]
    ).mean()
    missed_share = (
        missed["claimpilot_leakage_usd"] / missed["total_claim_amount"]
    ).mean()
    assert ref_share < missed_share


def test_unreferred_flagged_claims_leak_more_than_the_manual_arm(backtest_arms):
    """A flagged claim the engine failed to refer is charged in full.

    The manual arm's assumed miss rate is 0.55; a claim ClaimPilot did not
    refer is missed with certainty. The backtest must not flatter itself by
    charging the two the same.
    """
    flagged = backtest_arms[backtest_arms["fraud_reported"].astype(bool)]
    missed = flagged[~flagged["routed_to_siu"] & ~flagged["fast_tracked"]]
    assert len(missed) > 0
    assert (
        missed["claimpilot_leakage_usd"] > missed["baseline_leakage_usd"]
    ).all()


def test_fast_tracking_carries_an_overpayment_cost(backtest_arms):
    rate = float(config.assumption("leakage", "stp_overpayment_rate"))
    assert rate > 0
    fast = backtest_arms[backtest_arms["fast_tracked"]]
    assert (fast["claimpilot_leakage_usd"] > 0).all()
    unflagged_fast = fast[~fast["fraud_reported"].astype(bool)]
    expected = unflagged_fast["total_claim_amount"] * rate
    assert (expected - unflagged_fast["claimpilot_leakage_usd"]).abs().max() < 0.011


# ---------------------------------------------------------------------------
# Summary arithmetic
# ---------------------------------------------------------------------------


def test_totals_reconcile_with_the_row_level_columns(backtest_arms):
    s = summary(backtest_arms)
    for arm, prefix in (("manual", "baseline"), ("claimpilot", "claimpilot")):
        a = s["arms"][arm]
        assert a["total_lae_usd"] == pytest.approx(
            backtest_arms[f"{prefix}_lae_usd"].sum(), abs=0.01
        )
        assert a["total_leakage_usd"] == pytest.approx(
            backtest_arms[f"{prefix}_leakage_usd"].sum(), abs=0.01
        )
        assert a["total_cost_usd"] == pytest.approx(
            a["total_lae_usd"] + a["total_leakage_usd"], abs=0.01
        )


def test_fte_months_follow_from_total_touch_time(backtest_arms):
    hours = float(config.assumption("backtest", "adjuster_hours_per_month"))
    s = summary(backtest_arms)
    assert s["arms"]["claimpilot"]["adjuster_fte_months"] == pytest.approx(
        backtest_arms["claimpilot_handling_hours"].sum() / hours, abs=0.1
    )


def test_deltas_match_the_arm_totals(backtest_arms):
    s = summary(backtest_arms)
    d = deltas(backtest_arms)
    m, c = s["arms"]["manual"], s["arms"]["claimpilot"]
    assert d["lae_total_usd"] == pytest.approx(
        c["total_lae_usd"] - m["total_lae_usd"], abs=0.01
    )
    assert d["total_cost_pct"] == pytest.approx(
        (c["total_cost_usd"] - m["total_cost_usd"]) / m["total_cost_usd"], abs=1e-4
    )


def test_breakeven_is_reconstructable_independently(backtest_arms):
    """Recompute the break-even count from first principles and compare."""
    be = lae_breakeven(backtest_arms)
    siu_rate = float(
        config.assumption("costs", "siu_investigator_cost_per_hour_usd")
    )
    manual_total = backtest_arms["baseline_lae_usd"].sum()
    cp_ex_siu = (
        backtest_arms["claimpilot_lae_usd"]
        - backtest_arms["claimpilot_siu_hours"] * siu_rate
    ).sum()
    cost_each = (
        float(
            config.assumption(
                "handling_time_hours", "claimpilot", "siu_referral_extra"
            )
        )
        * siu_rate
    )
    assert be["breakeven_referrals"] == int((manual_total - cp_ex_siu) / cost_each)


def test_breakeven_verdict_matches_the_numbers(backtest_arms):
    be = lae_breakeven(backtest_arms)
    over = be["actual_referrals"] > be["breakeven_referrals"]
    assert ("costs more" in be["verdict"]) == over


def test_automation_saves_money_before_investigation_spend(backtest_arms):
    """Strip the SIU programme out and the automation must be cheaper.

    If this fails, the automation itself has stopped paying and the headline
    is no longer a question of referral volume.
    """
    assert lae_breakeven(backtest_arms)["automation_headroom_usd"] > 0


# ---------------------------------------------------------------------------
# SIU hit rate (measured, not assumed)
# ---------------------------------------------------------------------------


def test_siu_hit_rate_beats_the_portfolio_base_rate(backtest_arms):
    h = siu_hit_rate(backtest_arms)
    assert h["hit_rate_precision"] > h["portfolio_flag_rate"]


def test_siu_hit_rate_arithmetic_on_a_hand_built_frame():
    df = pd.DataFrame(
        {
            "routed_to_siu": [True, True, True, False, False],
            "fraud_reported": [True, True, False, True, False],
        }
    )
    h = siu_hit_rate(df)
    assert h["referred"] == 3
    assert h["hits"] == 2
    assert h["hit_rate_precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert h["flagged_caught_recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert h["flagged_not_referred"] == 1


def test_siu_hit_rate_survives_an_empty_referral_set():
    df = pd.DataFrame({"routed_to_siu": [False, False], "fraud_reported": [True, False]})
    h = siu_hit_rate(df)
    assert h["hit_rate_precision"] == 0.0
    assert h["referred"] == 0


# ---------------------------------------------------------------------------
# Honesty of the published object
# ---------------------------------------------------------------------------


def test_summary_is_labelled_as_simulated_and_not_evidence(backtest_arms):
    s = summary(backtest_arms)
    assert s["is_real"] is False
    assert "not evidence" in s["evidence_caveat"].lower()
    assert "paired" in s["design"]
    assert s["version"] == BACKTEST_VERSION


def test_module_docstring_warns_the_deltas_are_not_evidence():
    import claimpilot.backtest as mod

    assert "not evidence of a saving" in mod.__doc__


def test_siu_hit_rate_carries_the_reported_label_caveat(backtest_arms):
    caveat = siu_hit_rate(backtest_arms)["label_caveat"].lower()
    assert "reported flag" in caveat
    assert "adjudicated" in caveat


def test_leakage_config_admits_the_arms_are_not_like_for_like():
    note = str(config.assumption("leakage", "claimpilot_rate_source")).lower()
    assert "assumption" in note
    assert "derived" in note


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def test_comparison_frame_has_one_row_per_arm(backtest_arms):
    comp = comparison_frame(backtest_arms)
    assert list(comp["Arm"]) == ["All manual", "ClaimPilot"]


def test_by_route_covers_every_claim(backtest_arms):
    routes = by_route(backtest_arms)
    assert routes["Claims"].sum() == len(backtest_arms)
    assert routes["Share"].sum() == pytest.approx(1.0, abs=1e-6)


def test_by_route_shows_both_a_saving_and_an_overspend(backtest_arms):
    """The route table is the explanation, so it must carry both signs."""
    deltas_col = by_route(backtest_arms)["LAE delta ($)"]
    assert (deltas_col < 0).any(), "no route is cheaper than manual"
    assert (deltas_col > 0).any(), "no route costs more than manual"
