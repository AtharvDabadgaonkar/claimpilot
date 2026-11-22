"""The assumptions file is the single source of truth for simulated numbers."""

from __future__ import annotations

import yaml

from claimpilot import config


def test_assumptions_file_parses():
    data = config.load_assumptions()
    assert isinstance(data, dict) and data


def test_required_sections_present():
    data = config.load_assumptions()
    for section in (
        "meta",
        "dataset",
        "protected_and_excluded_fields",
        "deadlines",
        "handling_time_hours",
        "costs",
        "cycle_time_days",
        "leakage",
        "mock_policy",
        "narratives",
    ):
        assert section in data, f"assumptions.yaml missing section: {section}"


def test_seed_is_fixed_integer():
    assert isinstance(config.random_seed(), int)


def test_carrier_is_fictional():
    """No real insurer branding anywhere."""
    assert config.carrier_name() == "Northwind Mutual"


def test_dataset_license_recorded():
    assert config.assumption("dataset", "license") == "CC BY 4.0"
    assert config.assumption("dataset", "doi").startswith("10.17632/")
    assert config.dataset_attribution()


def test_dataset_limitations_documented():
    """Small-sample and inflated-fraud-rate caveats must be stated."""
    limits = " ".join(config.assumption("dataset", "limitations")).lower()
    assert "1,000" in limits or "1000" in limits
    assert "recalibrat" in limits
    assert "25%" in limits or "fraud rate" in limits


def test_protected_fields_excluded_from_models():
    excluded = config.excluded_model_fields()
    for field in ("insured_sex", "age", "insured_zip", "insured_education_level"):
        assert field in excluded, f"{field} must be excluded from model inputs"


def test_label_leaking_hobby_field_excluded():
    """insured_hobbies leaks the fraud label in this dataset."""
    assert "insured_hobbies" in config.excluded_model_fields()


def test_deadlines_labelled_illustrative():
    label = config.assumption("deadlines", "label").lower()
    assert "illustrative" in label


def test_simulated_sections_carry_a_label():
    """Any section of invented numbers must say so in the file itself."""
    data = config.load_assumptions()
    for section in (
        "handling_time_hours",
        "costs",
        "cycle_time_days",
        "leakage",
        "mock_policy",
        "mock_contact",
        "narratives",
    ):
        assert "SIMULATED" in str(data[section].get("label", "")).upper(), (
            f"{section} must be labelled SIMULATED"
        )


def test_disclaimer_wording_is_careful():
    """Never claim compliance or certification."""
    text = config.ASSUMPTIONS_PATH.read_text(encoding="utf-8").lower()
    assert "naic-compliant" not in text
    assert "certified" not in text
    assert config.DISCLAIMER.startswith("Demo project")


def test_yaml_is_not_executable():
    """safe_load only; no python object tags in the config."""
    raw = config.ASSUMPTIONS_PATH.read_text(encoding="utf-8")
    assert "!!python" not in raw
    yaml.safe_load(raw)
