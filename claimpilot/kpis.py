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


def extraction_metrics() -> dict[str, Any]:
    """Measured accuracy of the rule-based FNOL extractor.

    Scored against what each narrative *states*, not against the underlying
    row, so refusing to invent an unstated value counts as correct.
    """
    import pandas as pd

    from claimpilot.intake import extract_fnol

    narratives = pd.read_parquet(config.NARRATIVES_PARQUET)
    checks = {
        "loss_type": lambda e: e.loss_type.value,
        "loss_date": lambda e: e.loss_date.isoformat() if e.loss_date else None,
        "num_injured": lambda e: e.num_injured,
        "police_report_available": lambda e: e.police_report_available,
        "num_vehicles": lambda e: e.num_vehicles,
        "witnesses": lambda e: e.witnesses,
        "city": lambda e: e.city,
        "state": lambda e: e.state,
        "auto_make": lambda e: e.auto_make,
        "auto_year": lambda e: e.auto_year,
        "stated_amount_usd": lambda e: e.estimated_amount_usd,
    }
    correct = total = 0
    per_field: dict[str, float] = {}
    completeness: list[float] = []

    for _, row in narratives.iterrows():
        extracted = extract_fnol(row["narrative"])
        completeness.append(extracted.completeness)
        for field, getter in checks.items():
            truth = row[f"truth_{field}"]
            if (
                truth is not None
                and not isinstance(truth, bool | str)
                and pd.isna(truth)
            ):
                truth = None
            got = getter(extracted)
            if truth is None and got is None:
                ok = True
            elif truth is None or got is None:
                ok = False
            elif isinstance(truth, int | float) and not isinstance(truth, bool):
                ok = abs(float(truth) - float(got)) < 1
            else:
                ok = str(truth) == str(got)
            per_field[field] = per_field.get(field, 0) + int(ok)
            correct += int(ok)
            total += 1

    n = len(narratives)
    return {
        "narratives_scored": int(n),
        "fields_scored": int(total),
        "field_accuracy": round(correct / total, 4),
        "per_field_accuracy": {k: round(v / n, 4) for k, v in per_field.items()},
        "mean_completeness": round(float(sum(completeness) / n), 4),
        "extractor": "rules-v0.1 (regex, no LLM)",
        "is_real": True,
        "accuracy_caveat": (
            "NOT a production accuracy estimate. The narratives were generated "
            "from templates and the regular expressions were written against "
            "those templates, so this figure is close to circular. The "
            "hand-written paraphrase set in tests/fixtures was also used while "
            "tuning, so no genuinely held-out evaluation exists: all text in "
            "this project is synthetic. Read this as 'the regression suite "
            "passes', not as evidence the extractor would handle real "
            "claimant writing."
        ),
    }


def triage_routing_metrics() -> dict[str, Any]:
    """Portfolio-level outcomes of triage and routing."""
    from claimpilot.pipeline import load_scored

    scored = load_scored()
    n = len(scored)

    def share(mask) -> float:
        return round(float(mask.mean()), 4)

    return {
        "claims_routed": int(n),
        "stp_rate": share(scored["auto_approved"]),
        "siu_referral_rate": share(scored["destination"] == "siu_queue"),
        "senior_adjuster_rate": share(scored["destination"] == "senior_adjuster"),
        "adjuster_queue_rate": share(scored["destination"] == "adjuster_queue"),
        "requires_human_rate": share(scored["requires_human"]),
        "auto_denied_count": 0,
        "severity_bands": {
            k: round(v, 4)
            for k, v in scored["severity_band"].value_counts(normalize=True).items()
        },
        "complexity_bands": {
            k: round(v, 4)
            for k, v in scored["complexity_band"].value_counts(normalize=True).items()
        },
        "fraud_bands": {
            k: round(v, 4)
            for k, v in scored["fraud_band"].value_counts(normalize=True).items()
        },
        "coverage_confirmed_rate": share(scored["coverage_status"] == "Confirmed"),
        "coverage_not_in_force_count": int(
            (scored["coverage_status"] == "Not in force at date of loss").sum()
        ),
        "coverage_question_count": int(
            (scored["coverage_status"] == "Coverage question").sum()
        ),
        "is_real": True,
        "stp_caveat": (
            "The STP rate is low because of this dataset, not the design: most "
            "claims here are large (median ~$58k) and 24.7% carry the "
            "reported-fraud flag, so they fail the low-severity or low-fraud "
            "test. A realistic auto book would fast-track far more."
        ),
    }


def fraud_model_metrics() -> dict[str, Any]:
    from claimpilot.fraud_model import load_model

    bundle = load_model()
    return {**bundle["metrics"], "features_used": len(bundle["features"]), "is_real": True}


def reserve_model_metrics() -> dict[str, Any]:
    from claimpilot.reserve import load_model

    bundle = load_model()
    return {**bundle["metrics"], "features_used": len(bundle["features"]), "is_real": True}


def workbench_metrics() -> dict[str, Any]:
    """Queue, deadline and audit figures from the seeded demo state."""
    import pandas as pd

    from claimpilot.audit import seed_demo_log
    from claimpilot.pipeline import load_scored
    from claimpilot.workbench import add_deadline_status, build_queue, demo_today

    scored = load_scored()
    claims = pd.read_parquet(config.CLAIMS_PARQUET)
    deadlines = pd.read_parquet(config.SAMPLE_DIR / "deadlines.parquet")

    status = add_deadline_status(deadlines)
    queue = build_queue(scored, claims, deadlines)
    log = seed_demo_log(scored)

    def share(mask) -> float:
        return round(float(mask.mean()), 4)

    return {
        "demo_date": str(demo_today()),
        "adjuster_queue_size": int(
            (scored["destination"].isin(["adjuster_queue", "senior_adjuster"])).sum()
        ),
        "siu_queue_size": int((scored["destination"] == "siu_queue").sum()),
        "deadline_breached_rate": share(status["deadline_status"] == "Breached"),
        "deadline_at_risk_rate": share(status["deadline_status"] == "At risk"),
        "deadline_on_track_rate": share(status["deadline_status"] == "On track"),
        "acknowledge_met_rate": share(status["acknowledge_status"] == "Met"),
        "mean_priority_score": round(float(queue["priority_score"].mean()), 1),
        "audit_records_seeded": int(len(log)),
        "override_rate": round(log.override_rate, 4),
        "override_eligible_decisions": int(log.override_eligible_count),
        "is_real": False,
        "deadline_caveat": (
            "Deadlines are illustrative and not state-specific. The high "
            "breach rate reflects a 2015 dataset viewed from a fixed demo "
            "date, not an operational failure."
        ),
        "override_caveat": (
            "Override rate counts only decisions where the AI made a "
            "substantive call (fast-track or SIU referral). Claims routed to "
            "a queue for review are excluded: there is nothing to override."
        ),
    }


def build_metrics() -> dict[str, Any]:
    """Assemble every metric available at the current build phase."""
    metrics: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "assumptions_version": config.assumption("meta", "version"),
        "random_seed": config.random_seed(),
        "phases_complete": [
            "1-setup-and-data",
            "2-intake-and-triage",
            "3-workbench-and-siu",
        ],
        "disclaimer": config.DISCLAIMER,
        "reading_note": (
            "Numbers under 'dataset' and 'pii_masking' are measured. Numbers "
            "under 'simulated_layers' derive from assumptions in "
            "config/assumptions.yaml and are not observations of a real "
            "insurer's operations."
        ),
    }
    metrics.update(data_foundation_metrics())
    metrics["extraction"] = extraction_metrics()
    metrics["fraud_model"] = fraud_model_metrics()
    metrics["triage_routing"] = triage_routing_metrics()
    metrics["reserve_model"] = reserve_model_metrics()
    metrics["workbench"] = workbench_metrics()
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
