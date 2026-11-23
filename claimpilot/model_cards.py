"""Model inventory: one card per AI component.

Loaded from ``config/model_cards.yaml`` so the inventory is reviewable without
reading code, and so a reviewer can diff it. Live performance figures are
pulled from ``reports/metrics.json`` rather than copied into the card, so a
card can never quietly drift from the model it describes.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any

import yaml

from claimpilot import config

CARDS_PATH = config.CONFIG_DIR / "model_cards.yaml"


class RiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def label(self) -> str:
        return self.value.capitalize()

    @property
    def icon(self) -> str:
        return {"low": "🟢", "medium": "🟠", "high": "🔴"}[self.value]

    @property
    def oversight(self) -> str:
        return {
            "low": "Advisory only; a human acts on it or ignores it.",
            "medium": "Shapes routing or valuation, always under human review.",
            "high": "Can change what happens to a claim; requires the strongest controls.",
        }[self.value]


@dataclass
class ModelCard:
    id: str
    name: str
    version: str
    purpose: str
    technique: str
    risk_tier: RiskTier
    owner: str
    last_evaluated: date
    training_data: str
    inputs: list[str]
    outputs: list[str]
    excluded_inputs: list[str]
    human_oversight: str
    limitations: list[str]
    failure_modes: list[str]

    @property
    def is_high_risk(self) -> bool:
        return self.risk_tier is RiskTier.HIGH

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "risk_tier": self.risk_tier.value,
            "owner": self.owner,
            "last_evaluated": self.last_evaluated.isoformat(),
            "limitations_documented": len(self.limitations),
            "failure_modes_documented": len(self.failure_modes),
            "excluded_inputs_documented": len(self.excluded_inputs),
        }


def _clean(text: Any) -> str:
    return " ".join(str(text).split())


@functools.lru_cache(maxsize=1)
def load_cards() -> list[ModelCard]:
    if not CARDS_PATH.exists():
        raise FileNotFoundError(f"Missing model inventory at {CARDS_PATH}")

    with CARDS_PATH.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)

    cards: list[ModelCard] = []
    for raw in payload["cards"]:
        cards.append(
            ModelCard(
                id=str(raw["id"]),
                name=str(raw["name"]),
                version=str(raw["version"]),
                purpose=_clean(raw["purpose"]),
                technique=_clean(raw["technique"]),
                risk_tier=RiskTier(str(raw["risk_tier"]).lower()),
                owner=str(raw["owner"]),
                last_evaluated=_as_date(raw["last_evaluated"]),
                training_data=_clean(raw["training_data"]),
                inputs=[_clean(x) for x in raw.get("inputs", [])],
                outputs=[_clean(x) for x in raw.get("outputs", [])],
                excluded_inputs=[_clean(x) for x in raw.get("excluded_inputs", [])],
                human_oversight=_clean(raw["human_oversight"]),
                limitations=[_clean(x) for x in raw.get("limitations", [])],
                failure_modes=[_clean(x) for x in raw.get("failure_modes", [])],
            )
        )
    return cards


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    import pandas as pd

    return pd.to_datetime(value).date()


def card(card_id: str) -> ModelCard:
    for item in load_cards():
        if item.id == card_id:
            return item
    raise KeyError(f"No model card with id '{card_id}'")


def inventory_version() -> str:
    with CARDS_PATH.open("r", encoding="utf-8") as fh:
        return str(yaml.safe_load(fh)["version"])


def live_metrics_for(card_id: str) -> dict[str, Any]:
    """Pull the component's current measured figures from metrics.json.

    Kept out of the card itself so a card cannot drift from its model.
    """
    from claimpilot.kpis import load_metrics

    metrics = load_metrics()
    return {
        "fnol_extractor": lambda: {
            "Field accuracy (generated corpus)": f"{metrics['extraction']['field_accuracy']:.1%}",
            "Mean completeness": f"{metrics['extraction']['mean_completeness']:.0%}",
            "Narratives scored": f"{metrics['extraction']['narratives_scored']:,}",
        },
        "fraud_model": lambda: {
            "ROC AUC (5-fold CV)": f"{metrics['fraud_model']['roc_auc_cv_mean']:.3f}",
            "ROC AUC (test split)": f"{metrics['fraud_model']['roc_auc_test']:.3f}",
            "Precision @ threshold": f"{metrics['fraud_model']['precision_at_threshold']:.3f}",
            "Recall @ threshold": f"{metrics['fraud_model']['recall_at_threshold']:.3f}",
        },
        "reserve_model": lambda: {
            "R²": f"{metrics['reserve_model']['r2']:.3f}",
            "Mean absolute error": f"${metrics['reserve_model']['mae_usd']:,.0f}",
            "80% interval coverage": f"{metrics['reserve_model']['interval_coverage']:.1%}",
        },
        "copilot": lambda: {
            "Mode": "Rule-based (deterministic)",
            "PII masking detection rate": f"{metrics['pii_masking']['detection_rate']:.0%}",
            "Claim fields readable": "20 (allow-listed)",
        },
    }.get(card_id, dict)()


def inventory_summary() -> dict[str, Any]:
    cards = load_cards()
    return {
        "inventory_version": inventory_version(),
        "components": len(cards),
        "by_risk_tier": {
            tier.value: len([c for c in cards if c.risk_tier is tier])
            for tier in RiskTier
        },
        "all_have_an_owner": all(c.owner for c in cards),
        "all_have_limitations": all(c.limitations for c in cards),
        "all_have_failure_modes": all(c.failure_modes for c in cards),
        "cards": [c.to_dict() for c in cards],
    }
