"""Ingest and clean the Mendeley auto-claims dataset.

Source data (CC BY 4.0, see config/assumptions.yaml) is downloaded to
``data/raw/`` which is gitignored. This module cleans it and writes a small
Parquet sample to ``data/sample/`` which IS committed, so the deployed app
never needs to download anything at runtime.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from claimpilot import config

# Columns that arrive as "?" sentinels rather than empty cells.
_QUESTION_MARK_COLUMNS = [
    "collision_type",
    "property_damage",
    "police_report_available",
]

# "?" in collision_type is STRUCTURAL, not missing-at-random: a vehicle theft
# or a parked-car claim has no collision to classify. Flagging this separately
# keeps the triage model from treating it as an absent document.
_NO_COLLISION_INCIDENT_TYPES = {"Vehicle Theft", "Parked Car"}

# Market segment of the insured vehicle. Used ONLY as a fairness test group and
# as a severity feature — never as a proxy for the policyholder.
_VEHICLE_CATEGORY = {
    "Audi": "Luxury",
    "BMW": "Luxury",
    "Mercedes": "Luxury",
    "Accura": "Near-luxury",
    "Saab": "Near-luxury",
    "Volkswagen": "Near-luxury",
    "Chevrolet": "Mainstream",
    "Dodge": "Mainstream",
    "Ford": "Mainstream",
    "Honda": "Mainstream",
    "Jeep": "Mainstream",
    "Nissan": "Mainstream",
    "Suburu": "Mainstream",
    "Toyota": "Mainstream",
}

# US Census-style grouping of the incident states present in the data.
_REGION = {
    "NY": "Northeast",
    "PA": "Northeast",
    "SC": "South Atlantic",
    "NC": "South Atlantic",
    "VA": "South Atlantic",
    "WV": "South Atlantic",
    "OH": "Midwest",
}

_AGE_BANDS = [(18, 25), (26, 35), (36, 45), (46, 55), (56, 120)]


def age_band(age: float | int | None) -> str:
    """Bucket an age into a reporting band. Used for fairness testing only."""
    if age is None or pd.isna(age):
        return "Unknown"
    for low, high in _AGE_BANDS:
        if low <= age <= high:
            return "56+" if high == 120 else f"{low}-{high}"
    return "Unknown"


def download_raw(dest: Path | None = None, force: bool = False) -> Path:
    """Download the source CSV to ``data/raw/`` unless it is already present."""
    import urllib.request

    dest = dest or config.RAW_CSV
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    print(f"Downloading source dataset -> {dest}")
    urllib.request.urlretrieve(config.DATASET_URL, dest)  # noqa: S310
    return dest


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw frame into ClaimPilot's canonical claim table."""
    df = raw.copy()

    # The source CSV has a trailing all-null column from a stray comma.
    df = df.drop(columns=[c for c in df.columns if c.startswith("_c")], errors="ignore")
    df.columns = [c.strip().replace("-", "_") for c in df.columns]

    # Normalise "?" sentinels to real missing values.
    for col in _QUESTION_MARK_COLUMNS:
        if col in df.columns:
            df[col] = df[col].replace("?", pd.NA)

    # Structural vs. unknown missingness in collision_type.
    df["collision_not_applicable"] = df["incident_type"].isin(
        _NO_COLLISION_INCIDENT_TYPES
    )
    df["collision_type"] = df["collision_type"].where(
        ~df["collision_not_applicable"], "Not applicable"
    )

    # YES/NO/<missing> -> tri-state booleans. NA means "we don't know yet",
    # which the triage engine reads as a missing-document signal.
    for col in ("property_damage", "police_report_available"):
        df[col] = df[col].map({"YES": True, "NO": False}).astype("boolean")

    df["authorities_contacted"] = df["authorities_contacted"].fillna("None")

    # Dates.
    for col in ("policy_bind_date", "incident_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Target label. Kept as a boolean; it is a *reported* flag, not an
    # adjudicated finding (see dataset limitations in assumptions.yaml).
    df["fraud_reported"] = df["fraud_reported"].map({"Y": True, "N": False}).astype(bool)

    # Combined single limit "250/500" -> per-person / per-occurrence in dollars.
    csl = df["policy_csl"].astype(str).str.split("/", expand=True)
    df["csl_per_person_usd"] = pd.to_numeric(csl[0], errors="coerce") * 1000
    df["csl_per_occurrence_usd"] = pd.to_numeric(csl[1], errors="coerce") * 1000

    # Derived reporting dimensions (fairness test groups + severity features).
    df["region"] = df["incident_state"].map(_REGION).fillna("Other")
    df["vehicle_category"] = df["auto_make"].map(_VEHICLE_CATEGORY).fillna("Mainstream")
    df["age_band"] = df["age"].apply(age_band)
    df["vehicle_age_years"] = (
        df["incident_date"].dt.year - pd.to_numeric(df["auto_year"], errors="coerce")
    ).clip(lower=0)
    df["tenure_years"] = (df["months_as_customer"] / 12.0).round(2)

    # Complexity-relevant flags, derived from real fields.
    #
    # has_injury uses the reported bodily-injury COUNT only. It deliberately
    # does NOT use `injury_claim`: in this dataset injury_claim is a near-
    # mechanical ~12.5% slice of total_claim_amount and is non-zero for 97.5%
    # of claims, including claims with zero reported injuries. It is a cost-
    # allocation artifact, not evidence that anyone was hurt. Including it
    # pushed the injury rate to 99% and made the flag useless for triage.
    df["has_injury"] = df["bodily_injuries"].fillna(0) > 0

    # Kept for reserve modelling, explicitly named as an allocation share so it
    # is not mistaken for an injury indicator.
    df["injury_claim_share"] = (
        df["injury_claim"] / df["total_claim_amount"].replace(0, pd.NA)
    ).astype(float)
    df["multi_party"] = df["number_of_vehicles_involved"].fillna(1) > 1
    df["has_witnesses"] = df["witnesses"].fillna(0) > 0
    # Missing documentation: we genuinely do not know these yet.
    df["missing_police_report"] = df["police_report_available"].isna()
    df["missing_property_damage_info"] = df["property_damage"].isna()

    # Stable, non-identifying claim id. The source policy_number is retained
    # for the mock coverage lookup but is never a model input.
    df = df.reset_index(drop=True)
    df.insert(0, "claim_id", [f"CLM-2015-{i:05d}" for i in range(1, len(df) + 1)])

    return df


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(raw_path: Path, claims: pd.DataFrame) -> dict:
    """Record where the data came from and what we did to it."""
    ds = config.assumption("dataset")
    record = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {
            "name": ds["name"],
            "publisher": ds["source"],
            "doi": ds["doi"],
            "url": ds["url"],
            "license": ds["license"],
            "attribution": ds["attribution"],
        },
        "raw_file": {
            "filename": raw_path.name,
            "sha256": _sha256(raw_path),
            "bytes": raw_path.stat().st_size,
        },
        "processed": {
            "rows": int(len(claims)),
            "columns": int(claims.shape[1]),
            "fraud_reported_rate": round(float(claims["fraud_reported"].mean()), 4),
            "label_note": (
                "fraud_reported is a reported flag, not an adjudicated finding"
            ),
        },
        "simulated_components": [
            "FNOL narratives (generated from each row's real fields)",
            "mock policies and coverage terms",
            "fake policyholder contact details (PII-masking demo)",
            "adjuster handling times, cycle times and costs",
            "claim deadlines",
        ],
        "disclaimer": config.DISCLAIMER,
    }
    config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    config.PROVENANCE_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def run(force_download: bool = False) -> pd.DataFrame:
    """Full ingest: download -> clean -> write Parquet + provenance."""
    raw_path = download_raw(force=force_download)
    raw = pd.read_csv(raw_path)

    expected = int(config.assumption("dataset", "rows_expected"))
    if len(raw) != expected:
        print(f"  warning: expected {expected} rows, found {len(raw)}")

    claims = clean(raw)
    config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    claims.to_parquet(config.CLAIMS_PARQUET, index=False)
    write_provenance(raw_path, claims)

    size_kb = config.CLAIMS_PARQUET.stat().st_size / 1024
    print(
        f"Wrote {len(claims)} claims x {claims.shape[1]} cols -> "
        f"{config.CLAIMS_PARQUET.relative_to(config.ROOT)} ({size_kb:.0f} KB)"
    )
    return claims


def load_claims() -> pd.DataFrame:
    """Read the committed claims sample. Used by the app and the tests."""
    if not config.CLAIMS_PARQUET.exists():
        raise FileNotFoundError(
            f"{config.CLAIMS_PARQUET} is missing. Run `make data` to build it."
        )
    return pd.read_parquet(config.CLAIMS_PARQUET)
