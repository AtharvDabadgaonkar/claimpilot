"""Fraud model: feature governance and published performance."""

from __future__ import annotations

import numpy as np
import pytest

from claimpilot import config
from claimpilot.fraud_model import (
    MODEL_PATH,
    all_feature_names,
    assert_no_protected_features,
    build_design_matrix,
    load_model,
    predict_proba,
)

# --- Feature governance -------------------------------------------------------


def test_no_protected_attribute_reaches_the_model():
    assert_no_protected_features()


@pytest.mark.parametrize(
    "field",
    ["insured_sex", "age", "insured_zip", "insured_education_level",
     "insured_occupation", "insured_relationship", "insured_hobbies"],
)
def test_specific_protected_fields_are_absent(field):
    assert field not in all_feature_names()


def test_label_leaking_hobby_field_is_absent():
    """insured_hobbies would flatter the model; it leaks the label here."""
    assert "insured_hobbies" not in all_feature_names()


def test_geographic_and_wealth_proxies_are_absent():
    for field in ("incident_state", "incident_city", "auto_make",
                  "capital_gains", "capital_loss"):
        assert field not in all_feature_names()


def test_feature_list_is_an_allow_list_not_a_deny_list(claims):
    """Only named fields reach the model; everything else is dropped."""
    design = build_design_matrix(claims)
    assert set(design.columns) == set(all_feature_names())
    assert len(design.columns) < len(claims.columns)


def test_design_matrix_has_no_missing_categoricals(claims):
    design = build_design_matrix(claims)
    for col in config.assumption("fraud_model", "categorical_features"):
        assert design[col].notna().all()


def test_target_is_not_among_the_features():
    assert "fraud_reported" not in all_feature_names()


# --- Trained artifact ---------------------------------------------------------


def test_model_is_committed_and_small():
    assert MODEL_PATH.exists(), "run `make models`"
    size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
    assert size_mb < 5, f"model is {size_mb:.1f} MB; keep deployment light"


def test_saved_model_records_its_features_and_version():
    bundle = load_model()
    assert bundle["version"]
    assert set(bundle["features"]) == set(all_feature_names())
    assert bundle["threshold"] == config.assumption("fraud_model", "threshold")


def test_predictions_are_probabilities(claims):
    proba = predict_proba(claims.head(50))
    assert len(proba) == 50
    assert np.all((proba >= 0) & (proba <= 1))


def test_predictions_are_deterministic(claims):
    a = predict_proba(claims.head(20))
    b = predict_proba(claims.head(20))
    np.testing.assert_array_equal(a, b)


# --- Published performance ----------------------------------------------------


def test_model_beats_random():
    m = load_model()["metrics"]
    assert m["roc_auc_test"] > 0.6
    assert m["roc_auc_cv_mean"] > 0.6


def test_cross_validated_auc_is_published_alongside_the_test_split():
    """The single test split flatters the model; the CV figure must be shown."""
    m = load_model()["metrics"]
    assert "roc_auc_cv_mean" in m
    assert "roc_auc_cv_std" in m


def test_precision_and_recall_are_reported_at_the_chosen_threshold():
    m = load_model()["metrics"]
    assert 0 <= m["precision_at_threshold"] <= 1
    assert 0 <= m["recall_at_threshold"] <= 1
    assert m["threshold"] == config.assumption("fraud_model", "threshold")


def test_referral_rate_stays_within_assumed_siu_capacity():
    """The threshold is tuned against capacity, so it must respect it."""
    m = load_model()["metrics"]
    capacity = float(config.assumption("fraud_model", "siu_capacity_share"))
    assert m["flag_rate_at_threshold"] <= capacity + 0.02, (
        f"flag rate {m['flag_rate_at_threshold']:.1%} exceeds assumed SIU "
        f"capacity of {capacity:.0%}"
    )


def test_label_and_precision_caveats_travel_with_the_metrics():
    """Numbers must not be publishable without their caveats."""
    m = load_model()["metrics"]
    assert "reported flag" in m["label_caveat"]
    assert "optimistic" in m["precision_caveat"]


def test_threshold_rationale_is_documented():
    rationale = config.assumption("fraud_model", "threshold_rationale").lower()
    assert "capacity" in rationale
    assert "f1" in rationale
