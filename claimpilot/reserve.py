"""Reserve model: the expected ultimate cost of a claim, with a range.

Two design rules matter more here than accuracy:

1. **The number comes from this model, never from a language model.** An LLM
   may describe a reserve in a sentence, but it must never originate the
   figure. A test asserts the copilot's suggestion matches this model's output
   exactly.
2. **No leakage from the target.** ``injury_claim``, ``property_claim`` and
   ``vehicle_claim`` sum to ``total_claim_amount``, so including them would
   produce a near-perfect model that predicts nothing useful at first notice
   of loss, when those components do not yet exist. They are excluded.

The model predicts three quantiles, so the adjuster sees a range rather than
false precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from claimpilot import config

MODEL_PATH = config.MODELS_DIR / "reserve_model.joblib"
MODEL_VERSION = "reserve-gbq-v0.1"

_CATEGORICAL = (
    "incident_type",
    "collision_type",
    "incident_severity",
    "authorities_contacted",
)
_BOOLEAN = ("missing_police_report", "missing_property_damage_info")


def feature_names() -> list[str]:
    return list(config.assumption("reserve_model", "features"))


def assert_no_target_leakage() -> None:
    """Guard: no component of the target may be used as a feature."""
    leaks = set(config.assumption("reserve_model", "leakage_exclusions").keys())
    used = set(feature_names())
    if overlap := used & leaks:
        raise ValueError(
            f"Reserve model features leak the target: {sorted(overlap)}"
        )


def build_design_matrix(claims: pd.DataFrame) -> pd.DataFrame:
    assert_no_target_leakage()
    frame = claims[feature_names()].copy()
    for col in _BOOLEAN:
        if col in frame.columns:
            frame[col] = frame[col].astype(int)
    for col in _CATEGORICAL:
        if col in frame.columns:
            frame[col] = frame[col].astype(str).fillna("Unknown")
    for col in frame.columns:
        if col not in _CATEGORICAL:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _build_pipeline(quantile: float) -> Pipeline:
    numeric = [c for c in feature_names() if c not in _CATEGORICAL]
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(_CATEGORICAL),
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("pre", pre),
            (
                "reg",
                GradientBoostingRegressor(
                    loss="quantile",
                    alpha=quantile,
                    n_estimators=250,
                    max_depth=3,
                    learning_rate=0.06,
                    min_samples_leaf=20,
                    random_state=config.random_seed(),
                ),
            ),
        ]
    )


@dataclass
class ReserveSuggestion:
    """A reserve range produced by the model, never by an LLM."""

    point_usd: float
    low_usd: float
    high_usd: float
    model_version: str
    basis: str

    @property
    def range_width_usd(self) -> float:
        return self.high_usd - self.low_usd

    def to_dict(self) -> dict[str, Any]:
        return {
            "reserve_point_usd": round(self.point_usd, 2),
            "reserve_low_usd": round(self.low_usd, 2),
            "reserve_high_usd": round(self.high_usd, 2),
            "reserve_model_version": self.model_version,
            "reserve_basis": self.basis,
        }


@dataclass
class ReserveMetrics:
    mae_usd: float
    r2: float
    median_absolute_error_usd: float
    interval_coverage: float
    mean_interval_width_usd: float
    n_train: int
    n_test: int
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "mae_usd": round(self.mae_usd, 2),
            "median_absolute_error_usd": round(self.median_absolute_error_usd, 2),
            "r2": round(self.r2, 4),
            "interval_coverage": round(self.interval_coverage, 4),
            "interval_coverage_target": 0.80,
            "mean_interval_width_usd": round(self.mean_interval_width_usd, 2),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "leakage_note": (
                "injury/property/vehicle claim components are excluded; they sum "
                "to the target and are unknown at first notice of loss"
            ),
        }


def train(claims: pd.DataFrame | None = None, save: bool = True):
    """Fit the three quantile models and measure them on a held-out split."""
    from claimpilot.ingest import load_claims

    claims = load_claims() if claims is None else claims
    cfg = config.assumption("reserve_model")
    quantiles = cfg["quantiles"]

    X = build_design_matrix(claims)
    y = claims[cfg["target"]].astype(float).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(cfg["test_size"]), random_state=config.random_seed()
    )

    models: dict[str, Pipeline] = {}
    for name, q in (
        ("low", float(quantiles["low"])),
        ("point", float(quantiles["point"])),
        ("high", float(quantiles["high"])),
    ):
        pipeline = _build_pipeline(q)
        pipeline.fit(X_train, y_train)
        models[name] = pipeline

    point = models["point"].predict(X_test)
    low = models["low"].predict(X_test)
    high = models["high"].predict(X_test)

    # Does the predicted range actually contain the truth as often as it claims?
    inside = (y_test >= low) & (y_test <= high)

    metrics = ReserveMetrics(
        mae_usd=float(mean_absolute_error(y_test, point)),
        r2=float(r2_score(y_test, point)),
        median_absolute_error_usd=float(np.median(np.abs(y_test - point))),
        interval_coverage=float(inside.mean()),
        mean_interval_width_usd=float(np.mean(high - low)),
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        model_version=MODEL_VERSION,
    )

    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "models": models,
                "version": MODEL_VERSION,
                "features": feature_names(),
                "metrics": metrics.to_dict(),
            },
            MODEL_PATH,
            compress=3,
        )

    return models, metrics


def load_model() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} is missing. Run `make models` to train it."
        )
    return joblib.load(MODEL_PATH)


def suggest_reserve(
    claim: dict[str, Any] | pd.Series, bundle: dict[str, Any] | None = None
) -> ReserveSuggestion:
    """Reserve suggestion for one claim, from the model."""
    bundle = bundle or load_model()
    frame = build_design_matrix(pd.DataFrame([dict(claim)]))

    point = float(bundle["models"]["point"].predict(frame)[0])
    low = float(bundle["models"]["low"].predict(frame)[0])
    high = float(bundle["models"]["high"].predict(frame)[0])

    # Quantile models are fitted independently and can cross on unusual inputs.
    low, point, high = sorted((max(low, 0.0), max(point, 0.0), max(high, 0.0)))

    return ReserveSuggestion(
        point_usd=point,
        low_usd=low,
        high_usd=high,
        model_version=bundle["version"],
        basis=(
            "GradientBoosting quantile regression on incident and policy "
            "characteristics; claim amount components excluded as target leakage"
        ),
    )


def suggest_reserve_batch(
    claims: pd.DataFrame, bundle: dict[str, Any] | None = None
) -> pd.DataFrame:
    bundle = bundle or load_model()
    frame = build_design_matrix(claims)
    low = bundle["models"]["low"].predict(frame)
    point = bundle["models"]["point"].predict(frame)
    high = bundle["models"]["high"].predict(frame)

    stacked = np.sort(
        np.clip(np.vstack([low, point, high]), 0, None), axis=0
    )
    return pd.DataFrame(
        {
            "claim_id": claims["claim_id"].to_numpy(),
            "reserve_low_usd": np.round(stacked[0], 2),
            "reserve_point_usd": np.round(stacked[1], 2),
            "reserve_high_usd": np.round(stacked[2], 2),
            "reserve_model_version": bundle["version"],
        }
    )
