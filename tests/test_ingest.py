"""The processed sample must be clean, typed and reproducible."""

from __future__ import annotations

import pandas as pd

from claimpilot import config
from claimpilot.ingest import age_band


def test_sample_exists_and_is_committed_size():
    assert config.CLAIMS_PARQUET.exists()
    size_mb = config.CLAIMS_PARQUET.stat().st_size / 1024 / 1024
    assert size_mb < 25, f"sample is {size_mb:.1f} MB; keep it under 25 MB"


def test_row_count_matches_source(claims):
    assert len(claims) == config.assumption("dataset", "rows_expected")


def test_claim_ids_unique(claims):
    assert claims["claim_id"].is_unique
    assert claims["claim_id"].str.startswith("CLM-").all()


def test_question_mark_sentinels_removed(claims):
    """The raw CSV encodes missing values as '?'. None may survive."""
    for col in claims.columns:
        if claims[col].dtype == object:
            assert not (claims[col].astype(str) == "?").any(), f"'?' left in {col}"


def test_stray_trailing_column_dropped(claims):
    assert not [c for c in claims.columns if c.startswith("_c")]


def test_fraud_label_is_boolean(claims):
    assert claims["fraud_reported"].dtype == bool
    # Sanity-check the rate we report publicly.
    assert 0.20 < claims["fraud_reported"].mean() < 0.30


def test_dates_parsed(claims):
    for col in ("incident_date", "policy_bind_date"):
        assert pd.api.types.is_datetime64_any_dtype(claims[col])
        assert claims[col].notna().all()


def test_tri_state_document_flags(claims):
    """YES/NO/unknown must survive as a nullable boolean, not collapse to False."""
    for col in ("property_damage", "police_report_available"):
        assert str(claims[col].dtype) == "boolean"
        assert claims[col].isna().any(), f"{col} should retain unknowns"


def test_structural_vs_unknown_missing_collision(claims):
    """Theft and parked-car claims have no collision to classify."""
    no_collision = claims[claims["collision_not_applicable"]]
    assert len(no_collision) > 0
    assert (no_collision["collision_type"] == "Not applicable").all()
    assert set(no_collision["incident_type"]) <= {"Vehicle Theft", "Parked Car"}


def test_combined_single_limit_parsed(claims):
    assert (claims["csl_per_person_usd"] > 0).all()
    assert (claims["csl_per_occurrence_usd"] >= claims["csl_per_person_usd"]).all()


def test_derived_reporting_dimensions(claims):
    assert claims["region"].notna().all()
    assert claims["vehicle_category"].notna().all()
    assert claims["age_band"].notna().all()
    assert "Unknown" not in set(claims["age_band"])
    assert (claims["vehicle_age_years"] >= 0).all()


def test_age_band_boundaries():
    assert age_band(19) == "18-25"
    assert age_band(25) == "18-25"
    assert age_band(26) == "26-35"
    assert age_band(64) == "56+"
    assert age_band(None) == "Unknown"


def test_complexity_flags_consistent(claims):
    """An injury flag must agree with the underlying injury fields."""
    injured = claims[claims["bodily_injuries"] > 0]
    assert injured["has_injury"].all()
    multi = claims[claims["number_of_vehicles_involved"] > 1]
    assert multi["multi_party"].all()


def test_injury_flag_ignores_the_allocation_artifact(claims):
    """Regression: injury_claim is ~12.5% of the total on nearly every claim,
    including claims with zero reported injuries. Deriving has_injury from it
    inflated the injury rate to 99% and made the signal useless for triage.
    """
    assert (claims["has_injury"] == (claims["bodily_injuries"] > 0)).all()
    # Sanity: the flag must discriminate, not fire on almost everything.
    assert 0.4 < claims["has_injury"].mean() < 0.8
    # The artifact is retained, but under a name that cannot be misread.
    assert "injury_claim_share" in claims.columns
    assert (claims["injury_claim"] > 0).mean() > 0.9


def test_provenance_recorded():
    import json

    assert config.PROVENANCE_PATH.exists()
    rec = json.loads(config.PROVENANCE_PATH.read_text())
    assert rec["source"]["license"] == "CC BY 4.0"
    assert rec["source"]["doi"].startswith("10.17632/")
    assert len(rec["raw_file"]["sha256"]) == 64
    assert rec["simulated_components"], "must list what is simulated"
