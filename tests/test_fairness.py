"""Fairness testing: disparate impact across protected groups."""

from __future__ import annotations

import pandas as pd
import pytest

from claimpilot import config
from claimpilot.fairness import (
    assess,
    assess_all,
    build_fairness_frame,
    disparate_impact_ratio,
    summary,
)

# =============================================================================
# Hand-calculated examples.
# =============================================================================


def test_disparate_impact_matches_a_hand_calculated_example():
    """Required: the calculation matches a hand-worked example.

    Group A: 20 of 100 fast-tracked -> 0.20
    Group B: 10 of 100 fast-tracked -> 0.10
    Ratio = lowest / highest = 0.10 / 0.20 = 0.50, which is below 0.8.
    """
    frame = pd.DataFrame(
        {
            "group": ["A"] * 100 + ["B"] * 100,
            "stp": [True] * 20 + [False] * 80 + [True] * 10 + [False] * 90,
        }
    )
    result = assess(frame, "group", "stp", min_group_size=10)

    assert result.disparate_impact_ratio == pytest.approx(0.50)
    assert result.flagged is True
    assert result.lowest_group == "B"
    assert result.highest_group == "A"

    by_group = {g.group: g for g in result.groups}
    assert by_group["A"].rate == pytest.approx(0.20)
    assert by_group["B"].rate == pytest.approx(0.10)
    assert by_group["A"].favourable == 20
    assert by_group["B"].n == 100


def test_hand_calculated_example_just_above_the_threshold():
    """0.40 / 0.50 = 0.80 exactly, which is NOT below the threshold."""
    frame = pd.DataFrame(
        {
            "group": ["A"] * 100 + ["B"] * 100,
            "stp": [True] * 50 + [False] * 50 + [True] * 40 + [False] * 60,
        }
    )
    result = assess(frame, "group", "stp", min_group_size=10)
    assert result.disparate_impact_ratio == pytest.approx(0.80)
    assert result.flagged is False


def test_hand_calculated_three_group_example():
    """Ratio uses the extremes: 0.05 / 0.25 = 0.20."""
    frame = pd.DataFrame(
        {
            "group": ["A"] * 100 + ["B"] * 100 + ["C"] * 100,
            "stp": (
                [True] * 25 + [False] * 75
                + [True] * 15 + [False] * 85
                + [True] * 5 + [False] * 95
            ),
        }
    )
    result = assess(frame, "group", "stp", min_group_size=10)
    assert result.disparate_impact_ratio == pytest.approx(0.20)
    assert result.lowest_group == "C"
    assert result.highest_group == "A"


def test_equal_rates_give_a_ratio_of_one():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 50 + ["B"] * 50,
            "stp": ([True] * 10 + [False] * 40) * 2,
        }
    )
    result = assess(frame, "group", "stp", min_group_size=10)
    assert result.disparate_impact_ratio == pytest.approx(1.0)
    assert result.flagged is False


# =============================================================================
# Edge cases that must not silently produce a reassuring number.
# =============================================================================


def test_ratio_is_none_with_fewer_than_two_groups():
    assert disparate_impact_ratio({"A": 0.3}) is None
    assert disparate_impact_ratio({}) is None


def test_ratio_is_none_when_no_group_received_the_outcome():
    """0/0 is not 'perfectly fair'; it is unassessable."""
    assert disparate_impact_ratio({"A": 0.0, "B": 0.0}) is None

    frame = pd.DataFrame({"group": ["A"] * 50 + ["B"] * 50, "stp": [False] * 100})
    result = assess(frame, "group", "stp", min_group_size=10)
    assert result.disparate_impact_ratio is None
    assert result.flagged is False
    assert result.verdict == "Not assessable"


def test_small_groups_are_reported_but_excluded_from_the_ratio():
    """A group of 5 produces a wild rate that would dominate the comparison."""
    frame = pd.DataFrame(
        {
            "group": ["A"] * 100 + ["B"] * 100 + ["tiny"] * 5,
            "stp": [True] * 20 + [False] * 80
            + [True] * 20 + [False] * 80
            + [False] * 5,
        }
    )
    result = assess(frame, "group", "stp", min_group_size=30)
    assert "tiny" in result.excluded_small_groups
    assert result.disparate_impact_ratio == pytest.approx(1.0)
    # Still reported, so it cannot be quietly hidden.
    assert "tiny" in {g.group: g for g in result.groups}
    assert {g.group: g for g in result.groups}["tiny"].too_small is True


def test_threshold_comes_from_config():
    assert (
        float(config.assumption("fairness", "disparate_impact_threshold")) == 0.80
    )


def test_threshold_note_states_it_is_not_an_insurance_standard():
    note = str(config.assumption("fairness", "threshold_note")).lower()
    assert "employment" in note
    assert "not" in note


# =============================================================================
# Against the real portfolio.
# =============================================================================


def test_every_tested_attribute_is_a_protected_or_proxy_field(scored_portfolio, claims):
    """We test outcomes across exactly the fields the models may not see."""
    excluded = set(config.excluded_model_fields())
    tested = set(config.assumption("fairness", "test_groups"))
    # region, age_band and vehicle_category are derived from excluded fields.
    assert "insured_sex" in tested and "insured_sex" in excluded
    assert {"region", "age_band", "vehicle_category"} <= tested


def test_fairness_frame_joins_outcomes_to_attributes(scored_portfolio, claims):
    frame = build_fairness_frame(scored_portfolio, claims)
    assert len(frame) == len(claims)
    for column in ("stp", "siu", "region", "age_band", "vehicle_category"):
        assert column in frame.columns
    assert frame["stp"].dtype == bool


def test_all_results_are_produced(scored_portfolio, claims):
    results = assess_all(scored_portfolio, claims)
    attributes = {r.attribute for r in results}
    outcomes = {r.outcome for r in results}
    assert attributes == set(config.assumption("fairness", "test_groups"))
    assert outcomes == {"STP fast-track rate", "SIU referral rate"}


def test_group_rates_sum_back_to_the_portfolio(scored_portfolio, claims):
    """Sanity: the per-group counts must reconcile with the whole book."""
    results = assess_all(scored_portfolio, claims)
    stp_by_region = next(
        r for r in results if r.attribute == "region" and r.outcome.startswith("STP")
    )
    assert sum(g.n for g in stp_by_region.groups) == len(claims)
    assert sum(g.favourable for g in stp_by_region.groups) == int(
        scored_portfolio["auto_approved"].sum()
    )


def test_summary_reports_flags_honestly(scored_portfolio, claims):
    """The summary must not hide a failing test."""
    results = assess_all(scored_portfolio, claims)
    report = summary(results)
    assert report["tests_run"] == len(results)
    assert report["tests_flagged"] == len([r for r in results if r.flagged])
    assert len(report["flagged"]) == report["tests_flagged"]
    if report["tests_flagged"]:
        assert report["worst_ratio"] < report["threshold"]


def test_the_demo_surfaces_real_disparities(scored_portfolio, claims):
    """This portfolio genuinely fails several tests, and must keep saying so.

    If this ever passes cleanly, the fairness check has been weakened rather
    than the outcomes improved.
    """
    report = summary(assess_all(scored_portfolio, claims))
    assert report["tests_flagged"] > 0, (
        "the fairness check no longer flags anything; verify it was not weakened"
    )
