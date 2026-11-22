"""Configuration and path resolution for ClaimPilot.

Every simulated constant lives in ``config/assumptions.yaml``. Nothing in the
codebase should hard-code a business number; load it from here instead, so a
reviewer can audit every invented value in one file.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

# Repo root: this file is claimpilot/config.py, so two parents up.
ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = ROOT / "config"
ASSUMPTIONS_PATH = CONFIG_DIR / "assumptions.yaml"
ROUTING_RULES_PATH = CONFIG_DIR / "routing_rules.yaml"

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = DATA_DIR / "sample"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# Committed, processed artifacts the deployed app reads at runtime.
CLAIMS_PARQUET = SAMPLE_DIR / "claims.parquet"
NARRATIVES_PARQUET = SAMPLE_DIR / "narratives.parquet"
POLICIES_PARQUET = SAMPLE_DIR / "policies.parquet"
PROVENANCE_PATH = SAMPLE_DIR / "provenance.json"

RAW_CSV = RAW_DIR / "insurance_claims.csv"

DATASET_URL = (
    "https://data.mendeley.com/public-files/datasets/992mh7dk9y/files/"
    "a39df307-4896-4bb7-a4b8-f7d7fcdefed1/file_downloaded"
)

DISCLAIMER = (
    "Demo project using public and simulated data. Not a licensed insurance "
    "product, not legal or compliance advice."
)


@functools.lru_cache(maxsize=1)
def load_assumptions() -> dict[str, Any]:
    """Load and cache ``config/assumptions.yaml``."""
    if not ASSUMPTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing assumptions file at {ASSUMPTIONS_PATH}. "
            "It is required: every simulated constant is defined there."
        )
    with ASSUMPTIONS_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{ASSUMPTIONS_PATH} did not parse to a mapping.")
    return data


def assumption(*keys: str, default: Any = None) -> Any:
    """Read a nested assumption, e.g. ``assumption('costs', 'currency')``."""
    node: Any = load_assumptions()
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            if default is not None:
                return default
            raise KeyError(f"Assumption path not found: {'.'.join(keys)}")
        node = node[key]
    return node


def random_seed() -> int:
    """The single seed that makes all simulation reproducible."""
    return int(assumption("meta", "random_seed"))


def carrier_name() -> str:
    return str(assumption("meta", "carrier_name"))


def excluded_model_fields() -> list[str]:
    """Fields never passed to any model (protected characteristics + proxies)."""
    return list(
        assumption("protected_and_excluded_fields", "excluded_from_models")
    )


def dataset_attribution() -> str:
    return str(assumption("dataset", "attribution"))
