"""Reserve model: leakage guards, calibration, and the no-LLM-numbers rule."""

from __future__ import annotations

import numpy as np
import pytest

from claimpilot import config
from claimpilot.reserve import (
    MODEL_PATH,
    assert_no_target_leakage,
    build_design_matrix,
    feature_names,
    load_model,
    suggest_reserve,
    suggest_reserve_batch,
)

# --- Leakage ------------------------------------------------------------------


def test_no_target_leakage():
    assert_no_target_leakage()


@pytest.mark.parametrize(
    "field",
    ["total_claim_amount", "injury_claim", "property_claim", "vehicle_claim",
     "injury_claim_share"],
)
def test_claim_amount_components_are_not_features(field):
    """These sum to the target and do not exist at first notice of loss."""
    assert field not in feature_names()


def test_no_protected_attribute_in_reserve_features():
    excluded = set(config.excluded_model_fields())
    assert not (set(feature_names()) & excluded)


def test_design_matrix_contains_only_declared_features(claims):
    design = build_design_matrix(claims)
    assert set(design.columns) == set(feature_names())


# --- Artifact -----------------------------------------------------------------


def test_model_is_committed_and_small():
    assert MODEL_PATH.exists(), "run `make models`"
    assert MODEL_PATH.stat().st_size / 1024 / 1024 < 5


def test_bundle_holds_all_three_quantiles():
    bundle = load_model()
    assert set(bundle["models"]) == {"low", "point", "high"}
    assert bundle["version"]


# --- Suggestions --------------------------------------------------------------


def test_suggestion_is_a_coherent_range(claims):
    for i in range(25):
        suggestion = suggest_reserve(claims.iloc[i])
        assert suggestion.low_usd <= suggestion.point_usd <= suggestion.high_usd
        assert suggestion.low_usd >= 0
        assert suggestion.range_width_usd >= 0


def test_suggestion_is_deterministic(claims):
    a = suggest_reserve(claims.iloc[0])
    b = suggest_reserve(claims.iloc[0])
    assert a.point_usd == b.point_usd
    assert (a.low_usd, a.high_usd) == (b.low_usd, b.high_usd)


def test_suggestion_records_its_provenance(claims):
    suggestion = suggest_reserve(claims.iloc[0])
    assert suggestion.model_version
    assert "quantile" in suggestion.basis.lower()
    assert "leakage" in suggestion.basis.lower()


def test_more_severe_claims_get_larger_reserves(claims):
    """Sanity: the model must track severity, not produce noise."""
    batch = suggest_reserve_batch(claims).merge(
        claims[["claim_id", "incident_severity"]], on="claim_id"
    )
    means = batch.groupby("incident_severity")["reserve_point_usd"].mean()
    assert means["Total Loss"] > means["Trivial Damage"]
    assert means["Major Damage"] > means["Minor Damage"]


def test_batch_matches_single_claim_suggestions(claims):
    batch = suggest_reserve_batch(claims.head(10))
    for _, row in batch.iterrows():
        single = suggest_reserve(
            claims.loc[claims["claim_id"] == row["claim_id"]].iloc[0]
        )
        assert abs(single.point_usd - row["reserve_point_usd"]) < 0.01


# --- Published performance ----------------------------------------------------


def test_model_explains_a_meaningful_share_of_variance():
    assert load_model()["metrics"]["r2"] > 0.5


def test_prediction_interval_is_roughly_calibrated():
    """An 80% interval that contains 40% of outcomes would be misleading."""
    metrics = load_model()["metrics"]
    coverage = metrics["interval_coverage"]
    assert 0.70 <= coverage <= 0.90, (
        f"80% interval covers {coverage:.0%} of held-out claims"
    )


def test_leakage_caveat_travels_with_the_metrics():
    assert "excluded" in load_model()["metrics"]["leakage_note"]


# =============================================================================
# The reserve comes from the model, never from the LLM.
# =============================================================================


def test_copilot_reserve_equals_the_model_output(claims):
    """Required: the reserve suggestion comes from the model, not the LLM."""
    from claimpilot.copilot import build_briefing

    claim = claims.iloc[0]
    briefing = build_briefing(claim)
    model_suggestion = suggest_reserve(claim)
    assert briefing.reserve.point_usd == model_suggestion.point_usd
    assert briefing.reserve.low_usd == model_suggestion.low_usd
    assert briefing.reserve.high_usd == model_suggestion.high_usd


def test_reserve_is_unchanged_when_the_llm_output_changes(claims):
    """Swap the provider for one that talks nonsense; the number must hold."""
    from claimpilot.copilot import build_briefing
    from claimpilot.llm.base import LLMProvider

    class NonsenseProvider(LLMProvider):
        name, model, is_mock = "nonsense", "nonsense-1", True

        def _generate(self, prompt: str, **kwargs) -> str:
            return "The reserve should obviously be $999,999,999."

    claim = claims.iloc[3]
    briefing = build_briefing(claim, provider=NonsenseProvider())
    assert briefing.reserve.point_usd == suggest_reserve(claim).point_usd
    assert briefing.reserve.point_usd < 1_000_000


def test_reserve_note_states_the_rule():
    note = config.assumption("reserve_model", "reserve_note").lower()
    assert "never from the language model" in note


def test_predictions_are_finite(claims):
    batch = suggest_reserve_batch(claims)
    for col in ("reserve_low_usd", "reserve_point_usd", "reserve_high_usd"):
        assert np.isfinite(batch[col]).all()
