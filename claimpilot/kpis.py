"""Metric computation. Every published number is written here, then read from
``reports/metrics.json`` by the README and the app — never typed by hand.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from claimpilot import config


def data_foundation_metrics() -> dict[str, Any]:
    """Phase 1: facts about the data and the simulated layers."""
    from claimpilot.ingest import load_claims
    from claimpilot.llm.masking import mask_text

    claims = load_claims()
    narratives = pd.read_parquet(config.NARRATIVES_PARQUET)
    policies = pd.read_parquet(config.POLICIES_PARQUET)
    baseline = pd.read_parquet(config.SAMPLE_DIR / "baseline.parquet")

    # Measure the PII masker against the narratives that we know carry PII.
    with_pii = narratives[narratives["truth_contains_pii"]]
    masked = [mask_text(t) for t in with_pii["narrative"]]
    detected = sum(1 for m in masked if m.masked_count > 0)
    # Narratives we know carry NO PII: any hit there is a false positive.
    without_pii = narratives[~narratives["truth_contains_pii"]]
    fp = sum(
        1
        for t in without_pii["narrative"]
        if any(
            k in mask_text(t).kinds
            for k in ("EMAIL", "PHONE", "POLICY", "NAME", "SSN", "VIN")
        )
    )

    return {
        "dataset": {
            "source": config.assumption("dataset", "source"),
            "doi": config.assumption("dataset", "doi"),
            "license": config.assumption("dataset", "license"),
            "claims": int(len(claims)),
            "columns_after_cleaning": int(claims.shape[1]),
            "reported_fraud_rate": round(float(claims["fraud_reported"].mean()), 4),
            "median_total_claim_usd": float(claims["total_claim_amount"].median()),
            "mean_total_claim_usd": round(
                float(claims["total_claim_amount"].mean()), 2
            ),
            "injury_claim_rate": round(float(claims["has_injury"].mean()), 4),
            "multi_party_rate": round(float(claims["multi_party"].mean()), 4),
            "missing_police_report_rate": round(
                float(claims["missing_police_report"].mean()), 4
            ),
            "is_real": True,
        },
        "simulated_layers": {
            "narratives": int(len(narratives)),
            "narratives_mean_chars": int(narratives["char_count"].mean()),
            "narratives_with_pii": int(with_pii.shape[0]),
            "narratives_vague_amount": int(
                narratives["truth_stated_amount_usd"].isna().sum()
            ),
            "mock_policies": int(len(policies)),
            "policies_expired_at_loss": int(
                policies["expected_expired_at_loss"].sum()
            ),
            "policies_liability_only": int(policies["expected_liability_only"].sum()),
            "baseline_mean_handling_hours": round(
                float(baseline["baseline_handling_hours"].mean()), 2
            ),
            "baseline_mean_cycle_days": round(
                float(baseline["baseline_cycle_days"].mean()), 1
            ),
            "baseline_mean_lae_usd": round(
                float(baseline["baseline_lae_usd"].mean()), 2
            ),
            "is_real": False,
        },
        "pii_masking": {
            "narratives_known_to_contain_pii": int(with_pii.shape[0]),
            "detection_rate": round(detected / max(len(with_pii), 1), 4),
            "false_positive_count_on_clean_narratives": int(fp),
            "clean_narratives_checked": int(len(without_pii)),
            "method": "regex-based, demo grade",
            "is_real": True,
        },
        "governance": {
            "fields_excluded_from_models": config.excluded_model_fields(),
            "fields_excluded_count": len(config.excluded_model_fields()),
            "auto_deny_paths": 0,
            "auto_deny_note": "the system can auto-approve but never auto-deny",
        },
    }


def build_metrics() -> dict[str, Any]:
    """Assemble every metric available at the current build phase."""
    metrics: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "assumptions_version": config.assumption("meta", "version"),
        "random_seed": config.random_seed(),
        "phases_complete": ["1-setup-and-data"],
        "disclaimer": config.DISCLAIMER,
        "reading_note": (
            "Numbers under 'dataset' and 'pii_masking' are measured. Numbers "
            "under 'simulated_layers' derive from assumptions in "
            "config/assumptions.yaml and are not observations of a real "
            "insurer's operations."
        ),
    }
    metrics.update(data_foundation_metrics())
    return metrics


def write_metrics() -> dict[str, Any]:
    metrics = build_metrics()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_metrics() -> dict[str, Any]:
    if not config.METRICS_PATH.exists():
        raise FileNotFoundError(
            f"{config.METRICS_PATH} is missing. Run `make metrics` to build it."
        )
    return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
