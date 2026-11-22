"""No made-up metrics: every published number comes from reports/metrics.json."""

from __future__ import annotations

import json

from claimpilot import config
from claimpilot.kpis import build_metrics, load_metrics


def test_metrics_file_exists():
    assert config.METRICS_PATH.exists(), "run `make metrics`"


def test_metrics_is_valid_json():
    json.loads(config.METRICS_PATH.read_text())


def test_metrics_are_reproducible():
    """Rebuilding must reproduce the committed numbers (timestamp aside)."""
    on_disk = load_metrics()
    rebuilt = build_metrics()
    for key in ("dataset", "simulated_layers", "pii_masking", "governance"):
        assert on_disk[key] == rebuilt[key], f"{key} drifted from the committed file"


def test_real_and_simulated_numbers_are_distinguished():
    m = load_metrics()
    assert m["dataset"]["is_real"] is True
    assert m["simulated_layers"]["is_real"] is False
    assert "reading_note" in m


def test_dataset_metrics_match_the_sample(claims):
    m = load_metrics()["dataset"]
    assert m["claims"] == len(claims)
    assert abs(m["reported_fraud_rate"] - claims["fraud_reported"].mean()) < 1e-6
    assert m["license"] == "CC BY 4.0"


def test_pii_masking_is_measured_not_claimed():
    m = load_metrics()["pii_masking"]
    assert m["narratives_known_to_contain_pii"] > 100
    assert m["detection_rate"] == 1.0
    assert m["false_positive_count_on_clean_narratives"] == 0
    assert m["clean_narratives_checked"] > 100


def test_governance_records_no_auto_deny_path():
    """Hard product rule: auto-approve yes, auto-deny never."""
    g = load_metrics()["governance"]
    assert g["auto_deny_paths"] == 0
    assert "never auto-deny" in g["auto_deny_note"]


def test_excluded_fields_recorded_in_metrics():
    g = load_metrics()["governance"]
    assert g["fields_excluded_count"] == len(g["fields_excluded_from_models"])
    assert "insured_sex" in g["fields_excluded_from_models"]


def test_seed_recorded_for_reproducibility():
    assert load_metrics()["random_seed"] == config.random_seed()
