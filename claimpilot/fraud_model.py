"""Fraud-signal model.

Trained on the dataset's ``fraud_reported`` flag. Two things to hold in mind
when reading any number this module produces:

* ``fraud_reported`` is a **reported** flag, not an adjudicated fraud finding.
  The model learns which claims got referred, inheriting whatever bias the
  original referral process had.
* The portfolio fraud rate here is ~25%, roughly 10-25x a real auto book, so
  precision in particular is wildly optimistic relative to production.

Features come from an explicit allow-list in ``config/assumptions.yaml``.
A field must be named there to reach the model; everything else is excluded
by default, which is the safer direction for a protected-attribute mistake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from claimpilot import config

MODEL_PATH = config.MODELS_DIR / "fraud_model.joblib"
MODEL_VERSION = "fraud-hgb-v0.1"


def feature_columns() -> dict[str, list[str]]:
    cfg = config.assumption("fraud_model")
    return {
        "numeric": list(cfg["numeric_features"]),
        "categorical": list(cfg["categorical_features"]),
        "boolean": list(cfg["boolean_features"]),
    }


def all_feature_names() -> list[str]:
    cols = feature_columns()
    return cols["numeric"] + cols["categorical"] + cols["boolean"]


def assert_no_protected_features() -> None:
    """Guard: no protected characteristic or proxy may reach the model."""
    excluded = set(config.excluded_model_fields())
    extra = set(config.assumption("fraud_model", "additional_exclusions").keys())
    used = set(all_feature_names())
    leaked = used & (excluded | extra)
    if leaked:
        raise ValueError(
            f"Protected or excluded fields present in model features: {sorted(leaked)}"
        )


def build_design_matrix(claims: pd.DataFrame) -> pd.DataFrame:
    """Select and type the allow-listed features."""
    assert_no_protected_features()
    cols = feature_columns()
    frame = claims[all_feature_names()].copy()
    for col in cols["boolean"]:
        frame[col] = frame[col].astype(int)
    for col in cols["categorical"]:
        frame[col] = frame[col].astype(str).fillna("Unknown")
    for col in cols["numeric"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def build_pipeline() -> Pipeline:
    cols = feature_columns()
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                cols["numeric"],
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cols["categorical"],
            ),
            ("bool", "passthrough", cols["boolean"]),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=4,
                    max_iter=200,
                    learning_rate=0.06,
                    min_samples_leaf=25,
                    l2_regularization=1.0,
                    random_state=config.random_seed(),
                ),
            ),
        ]
    )


@dataclass
class FraudModelMetrics:
    """Everything published about model performance."""

    roc_auc_test: float
    roc_auc_cv_mean: float
    roc_auc_cv_std: float
    average_precision_test: float
    brier_test: float
    threshold: float
    precision_at_threshold: float
    recall_at_threshold: float
    flag_rate_at_threshold: float
    base_rate: float
    n_train: int
    n_test: int
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "roc_auc_test": round(self.roc_auc_test, 4),
            "roc_auc_cv_mean": round(self.roc_auc_cv_mean, 4),
            "roc_auc_cv_std": round(self.roc_auc_cv_std, 4),
            "average_precision_test": round(self.average_precision_test, 4),
            "brier_test": round(self.brier_test, 4),
            "threshold": self.threshold,
            "precision_at_threshold": round(self.precision_at_threshold, 4),
            "recall_at_threshold": round(self.recall_at_threshold, 4),
            "flag_rate_at_threshold": round(self.flag_rate_at_threshold, 4),
            "base_rate": round(self.base_rate, 4),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "label_caveat": (
                "fraud_reported is a reported flag, not an adjudicated finding"
            ),
            "precision_caveat": (
                "the ~25% base rate here is far above a real auto portfolio, so "
                "precision is optimistic and would fall sharply in production"
            ),
        }


def train(claims: pd.DataFrame | None = None, save: bool = True):
    """Train the fraud model and compute its published metrics."""
    from claimpilot.ingest import load_claims

    claims = load_claims() if claims is None else claims
    cfg = config.assumption("fraud_model")

    X = build_design_matrix(claims)
    y = claims["fraud_reported"].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(cfg["test_size"]),
        random_state=config.random_seed(),
        stratify=y,
    )

    pipeline = build_pipeline()

    # Cross-validated AUC on the training split, to show the test figure is
    # not a lucky split.
    cv = StratifiedKFold(
        n_splits=int(cfg["cv_folds"]), shuffle=True, random_state=config.random_seed()
    )
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")

    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]

    threshold = float(cfg["threshold"])
    predicted = (proba >= threshold).astype(int)

    metrics = FraudModelMetrics(
        roc_auc_test=float(roc_auc_score(y_test, proba)),
        roc_auc_cv_mean=float(np.mean(cv_scores)),
        roc_auc_cv_std=float(np.std(cv_scores)),
        average_precision_test=float(average_precision_score(y_test, proba)),
        brier_test=float(brier_score_loss(y_test, proba)),
        threshold=threshold,
        precision_at_threshold=float(precision_score(y_test, predicted, zero_division=0)),
        recall_at_threshold=float(recall_score(y_test, predicted, zero_division=0)),
        flag_rate_at_threshold=float(predicted.mean()),
        base_rate=float(y.mean()),
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        model_version=MODEL_VERSION,
    )

    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": pipeline,
                "version": MODEL_VERSION,
                "features": all_feature_names(),
                "threshold": threshold,
                "metrics": metrics.to_dict(),
            },
            MODEL_PATH,
            compress=3,
        )

    return pipeline, metrics


def load_model() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} is missing. Run `make models` to train it."
        )
    return joblib.load(MODEL_PATH)


def predict_proba(claims: pd.DataFrame, bundle: dict[str, Any] | None = None) -> np.ndarray:
    """Fraud probability for each claim."""
    bundle = bundle or load_model()
    X = build_design_matrix(claims)
    return bundle["pipeline"].predict_proba(X)[:, 1]
