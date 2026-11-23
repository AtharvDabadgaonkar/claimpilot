"""Adjuster queue: deadline status and priority ordering.

Priority is a transparent weighted sum, not a model. An adjuster has to be
able to see why a claim is at the top of their list, and a manager has to be
able to change the weights without a retrain. Weights live in
``config/assumptions.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any

import pandas as pd

from claimpilot import config


class DeadlineStatus(str, Enum):
    ON_TRACK = "On track"
    AT_RISK = "At risk"
    BREACHED = "Breached"

    @property
    def icon(self) -> str:
        return {"On track": "🟢", "At risk": "🟠", "Breached": "🔴"}[self.value]


def demo_today() -> date:
    """Fixed 'today' for the demo.

    The dataset's losses fall in Jan-Mar 2015. Using the real current date
    would mark every claim a decade overdue and make the tracker useless.
    """
    return pd.to_datetime(config.assumption("workbench", "demo_today")).date()


def deadline_status(
    due: Any, today: date | None = None, window_days: int | None = None
) -> DeadlineStatus:
    """Classify one deadline against the demo date."""
    today = today or demo_today()
    due_date = pd.to_datetime(due, errors="coerce")
    if pd.isna(due_date):
        return DeadlineStatus.ON_TRACK
    due_date = due_date.date()

    if today > due_date:
        return DeadlineStatus.BREACHED

    threshold = float(config.assumption("deadlines", "at_risk_threshold_pct")) / 100.0
    window = window_days or int(config.assumption("deadlines", "decide_days"))
    days_left = (due_date - today).days
    if days_left <= window * (1 - threshold):
        return DeadlineStatus.AT_RISK
    return DeadlineStatus.ON_TRACK


def add_deadline_status(
    deadlines: pd.DataFrame, today: date | None = None
) -> pd.DataFrame:
    """Status for the acknowledge, decide and pay deadlines of every claim."""
    today = today or demo_today()
    frame = deadlines.copy()

    for name, window_key in (
        ("acknowledge", "acknowledge_days"),
        ("decide", "decide_days"),
        ("pay", "pay_days"),
    ):
        window = int(config.assumption("deadlines", window_key))
        frame[f"{name}_status"] = frame[f"{name}_due"].apply(
            lambda d, w=window: deadline_status(d, today, w).value
        )

    # An obligation already MET is not a pending breach. Digital FNOL
    # acknowledges on submission, so the acknowledge clock is satisfied and
    # drops out of the queue signal. Without this every claim in a 2015
    # dataset reads as "Breached" and the tracker conveys nothing.
    if "acknowledge_met" in frame.columns:
        frame.loc[frame["acknowledge_met"].fillna(False), "acknowledge_status"] = "Met"

    # Headline status follows the next OPEN obligation: decide, then pay.
    order = {s.value: i for i, s in enumerate(DeadlineStatus)}
    frame["deadline_status"] = frame[["decide_status", "pay_status"]].apply(
        lambda row: max(row, key=lambda v: order[v]), axis=1
    )

    frame["days_to_decide_due"] = (
        pd.to_datetime(frame["decide_due"]).dt.date.apply(lambda d: (d - today).days)
    )
    frame["age_days"] = (
        pd.to_datetime(frame["reported_date"]).dt.date.apply(lambda d: (today - d).days)
    )
    return frame


@dataclass
class PriorityScore:
    value: float
    reasons: list[str]


def score_priority(row: dict[str, Any]) -> PriorityScore:
    """Weighted priority for one claim, with the drivers spelled out."""
    weights = config.assumption("workbench", "priority_weights")
    total = 0.0
    reasons: list[str] = []

    def add(points: float, text: str) -> None:
        nonlocal total
        if points:
            total += points
            reasons.append(f"{text} (+{points:g})")

    if row.get("severity_band") == "High":
        add(float(weights["severity_high"]), "High severity")
    elif row.get("severity_band") == "Medium":
        add(float(weights["severity_medium"]), "Medium severity")

    status = row.get("deadline_status")
    if status == DeadlineStatus.BREACHED.value:
        add(float(weights["deadline_breached"]), "Deadline breached")
    elif status == DeadlineStatus.AT_RISK.value:
        add(float(weights["deadline_at_risk"]), "Deadline at risk")

    if row.get("fraud_band") == "High":
        add(float(weights["fraud_high"]), "High fraud signal")
    elif row.get("fraud_band") == "Medium":
        add(float(weights["fraud_medium"]), "Medium fraud signal")

    if row.get("complexity_band") == "High":
        add(float(weights["complexity_high"]), "High complexity")

    if row.get("coverage_status") != "Confirmed":
        add(float(weights["coverage_problem"]), "Open coverage issue")

    age = float(row.get("age_days") or 0)
    if age > 0:
        points = round(age * float(weights["age_days_multiplier"]), 1)
        add(points, f"{int(age)} days since reported")

    return PriorityScore(value=round(total, 1), reasons=reasons)


def build_queue(
    scored: pd.DataFrame,
    claims: pd.DataFrame,
    deadlines: pd.DataFrame,
    destinations: list[str] | None = None,
    today: date | None = None,
) -> pd.DataFrame:
    """The prioritised work queue, highest priority first."""
    dl = add_deadline_status(deadlines, today)

    queue = (
        scored.merge(
            dl[
                [
                    "claim_id",
                    "deadline_status",
                    "acknowledge_status",
                    "decide_status",
                    "pay_status",
                    "decide_due",
                    "days_to_decide_due",
                    "age_days",
                ]
            ],
            on="claim_id",
            how="left",
        )
        .merge(
            claims[
                [
                    "claim_id",
                    "incident_type",
                    "incident_severity",
                    "total_claim_amount",
                    "incident_date",
                    "bodily_injuries",
                    "number_of_vehicles_involved",
                    "witnesses",
                    "has_injury",
                    "multi_party",
                    "missing_police_report",
                    "missing_property_damage_info",
                ]
            ],
            on="claim_id",
            how="left",
        )
    )

    if destinations is not None:
        queue = queue[queue["destination"].isin(destinations)]

    scores = [score_priority(row) for _, row in queue.iterrows()]
    queue = queue.assign(
        priority_score=[s.value for s in scores],
        priority_reasons=[s.reasons for s in scores],
    )
    return queue.sort_values("priority_score", ascending=False).reset_index(drop=True)
