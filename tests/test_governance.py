"""Model inventory, monitoring signals and the governance document."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from claimpilot import config
from claimpilot.audit import AuditLog, DecisionType, seed_demo_log
from claimpilot.model_cards import (
    CARDS_PATH,
    RiskTier,
    card,
    inventory_summary,
    inventory_version,
    live_metrics_for,
    load_cards,
)
from claimpilot.monitoring import (
    SignalStatus,
    all_signals,
    band_mix_over_time,
    drift_signal,
    override_signal,
    population_stability_index,
    scores_over_time,
)

GOVERNANCE_DOC = config.ROOT / "docs" / "AI_GOVERNANCE.md"


# =============================================================================
# Model inventory
# =============================================================================


def test_inventory_file_exists_and_is_versioned():
    assert CARDS_PATH.exists()
    assert inventory_version().startswith("model-inventory-")


def test_every_ai_component_has_a_card():
    """Required: a model card for each AI component."""
    ids = {c.id for c in load_cards()}
    assert ids == {"fnol_extractor", "fraud_model", "reserve_model", "copilot"}


@pytest.mark.parametrize(
    "card_id", ["fnol_extractor", "fraud_model", "reserve_model", "copilot"]
)
def test_card_documents_everything_the_bulletin_expects(card_id):
    item = card(card_id)
    assert item.purpose and len(item.purpose) > 40
    assert item.technique
    assert item.owner
    assert item.training_data
    assert item.human_oversight
    assert item.inputs
    assert item.excluded_inputs
    assert item.limitations, "a card with no limitations is not an honest card"
    assert item.failure_modes
    assert isinstance(item.risk_tier, RiskTier)
    assert item.last_evaluated


def test_risk_tiers_are_assigned_sensibly():
    """The fraud model moves claims into an SIU queue, so it is high risk."""
    assert card("fraud_model").risk_tier is RiskTier.HIGH
    assert card("fraud_model").is_high_risk
    # The copilot is advisory only.
    assert card("copilot").risk_tier is RiskTier.LOW


def test_fraud_card_documents_the_label_and_base_rate_caveats():
    limitations = " ".join(card("fraud_model").limitations).lower()
    assert "reported" in limitations
    assert "adjudicated" in limitations
    assert "recalibration" in limitations or "recalibrat" in limitations


def test_extractor_card_admits_the_accuracy_is_near_circular():
    limitations = " ".join(card("fnol_extractor").limitations).lower()
    assert "circular" in limitations
    assert "held-out" in limitations


def test_reserve_card_documents_target_leakage():
    limitations = " ".join(card("reserve_model").limitations).lower()
    assert "leakage" in limitations


def test_cards_record_excluded_protected_inputs():
    fraud = " ".join(card("fraud_model").excluded_inputs)
    for field in ("insured_sex", "age", "insured_zip", "insured_hobbies"):
        assert field in fraud


def test_copilot_card_states_it_cannot_decide():
    oversight = card("copilot").human_oversight.lower()
    assert "cannot" in oversight


def test_card_versions_match_the_live_code():
    """A card must not drift from the model it describes."""
    from claimpilot.fraud_model import MODEL_VERSION as FRAUD_VERSION
    from claimpilot.reserve import MODEL_VERSION as RESERVE_VERSION

    assert card("fraud_model").version == FRAUD_VERSION
    assert card("reserve_model").version == RESERVE_VERSION


def test_live_metrics_are_read_not_copied():
    """Performance figures come from metrics.json, not the card text."""
    from claimpilot.kpis import load_metrics

    metrics = load_metrics()
    shown = live_metrics_for("fraud_model")
    assert f"{metrics['fraud_model']['roc_auc_cv_mean']:.3f}" in shown.values()
    assert live_metrics_for("reserve_model")
    assert live_metrics_for("fnol_extractor")


def test_inventory_summary_is_complete():
    report = inventory_summary()
    assert report["components"] == 4
    assert report["all_have_an_owner"]
    assert report["all_have_limitations"]
    assert report["all_have_failure_modes"]
    assert sum(report["by_risk_tier"].values()) == 4


# =============================================================================
# Monitoring
# =============================================================================


def test_psi_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    psi = population_stability_index(rng.normal(size=5000), rng.normal(size=5000))
    assert psi < 0.02


def test_psi_is_large_for_a_shifted_distribution():
    rng = np.random.default_rng(0)
    psi = population_stability_index(
        rng.normal(0, 1, 5000), rng.normal(1.5, 1, 5000)
    )
    assert psi > 0.25


def test_psi_is_finite_with_empty_bins():
    """An empty bin must not send PSI to infinity."""
    baseline = np.linspace(0, 1, 1000)
    shifted = np.linspace(0.9, 1.0, 1000)
    psi = population_stability_index(baseline, shifted)
    assert np.isfinite(psi)


def test_psi_handles_empty_input():
    assert np.isnan(population_stability_index(np.array([]), np.array([1.0])))


def test_override_signal_needs_enough_decisions():
    log = AuditLog()
    for _ in range(3):
        log.record_decision(
            claim_id="C", ai_recommendation="siu_queue",
            decision=DecisionType.ESCALATE_SIU, reason_code="ESC_OTHER",
        )
    signal = override_signal(log)
    assert signal.status is SignalStatus.INSUFFICIENT


def test_override_signal_alerts_when_humans_often_disagree():
    log = AuditLog()
    for _ in range(30):
        log.record_decision(
            claim_id="C", ai_recommendation="siu_queue", decision=DecisionType.APPROVE
        )
    signal = override_signal(log)
    assert signal.status is SignalStatus.ALERT
    assert signal.value == 1.0


def test_near_zero_override_rate_is_also_a_warning():
    """Total agreement may mean rubber-stamping, not a perfect model."""
    log = AuditLog()
    for _ in range(60):
        log.record_decision(
            claim_id="C", ai_recommendation="siu_queue",
            decision=DecisionType.ESCALATE_SIU, reason_code="ESC_SUSPECTED_FRAUD",
        )
    signal = override_signal(log)
    assert signal.value == 0.0
    assert signal.status is SignalStatus.WARN
    assert "rubber-stamping" in signal.detail


def test_seeded_demo_log_sits_in_the_healthy_range(scored_portfolio):
    signal = override_signal(seed_demo_log(scored_portfolio))
    assert signal.status is SignalStatus.OK


def test_drift_signal_runs_on_the_portfolio(scored_portfolio, claims):
    signal = drift_signal(scored_portfolio, claims)
    assert signal.value is not None
    assert signal.status in set(SignalStatus)
    assert "too few" in signal.detail or "not live scores" in signal.detail or "far too few" in signal.detail


def test_drift_signal_always_carries_its_caveat(scored_portfolio, claims):
    """The number must never be publishable without the warning."""
    assert "too few" in drift_signal(scored_portfolio, claims).detail


def test_drift_needs_enough_claims(claims):
    tiny = pd.DataFrame({"claim_id": claims["claim_id"].head(10), "fraud_probability": 0.3})
    signal = drift_signal(tiny, claims)
    assert signal.status is SignalStatus.INSUFFICIENT


def test_scores_over_time_produces_a_trend(scored_portfolio, claims):
    trend = scores_over_time(scored_portfolio, claims)
    assert len(trend) > 3
    for column in ("claims", "mean_fraud_probability", "stp_rate", "siu_rate"):
        assert column in trend.columns
    assert trend["claims"].sum() == len(scored_portfolio)


def test_band_mix_shares_sum_to_one(scored_portfolio, claims):
    mix = band_mix_over_time(scored_portfolio, claims)
    totals = mix.groupby("period")["share"].sum()
    assert np.allclose(totals.to_numpy(), 1.0)


def test_all_signals_returns_both(scored_portfolio, claims):
    signals = all_signals(seed_demo_log(scored_portfolio), scored_portfolio, claims)
    assert {s.name for s in signals} == {"Override rate", "Score drift (PSI)"}


# =============================================================================
# The governance document
# =============================================================================


def test_governance_doc_exists():
    assert GOVERNANCE_DOC.exists()


def test_governance_doc_covers_every_required_section():
    text = GOVERNANCE_DOC.read_text(encoding="utf-8").lower()
    for topic in (
        "programme summary", "risk tier", "human-in-the-loop", "fairness",
        "audit", "monitoring", "privacy", "beyond this demo",
    ):
        assert topic in text, f"AI_GOVERNANCE.md is missing: {topic}"


def test_governance_doc_uses_careful_compliance_wording():
    text = GOVERNANCE_DOC.read_text(encoding="utf-8").lower()
    assert "designed around" in text
    assert "not compliant with, nor certified against" in text
    assert "naic-compliant" not in text


def test_governance_doc_carries_the_disclaimer():
    text = " ".join(GOVERNANCE_DOC.read_text(encoding="utf-8").split())
    assert "Not a licensed insurance product" in text


def test_governance_doc_states_the_no_auto_deny_rule():
    text = GOVERNANCE_DOC.read_text(encoding="utf-8").lower()
    assert "never auto-deny" in text or "can never auto-deny" in text


def test_governance_doc_admits_the_fairness_failure():
    """The doc must report the failing tests, not gloss over them."""
    text = GOVERNANCE_DOC.read_text(encoding="utf-8").lower()
    assert "fails 5 of 8" in text
    assert "0.355" in text


def test_governance_doc_qualifies_the_fairness_threshold():
    text = GOVERNANCE_DOC.read_text(encoding="utf-8").lower()
    assert "employment" in text
    assert "not an insurance standard" in text


def test_governance_doc_lists_what_a_real_insurer_needs():
    text = GOVERNANCE_DOC.read_text(encoding="utf-8").lower()
    for requirement in (
        "state", "independent model validation", "recalibration", "filings",
    ):
        assert requirement in text
