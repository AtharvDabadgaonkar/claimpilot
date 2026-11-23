"""Ongoing monitoring: override rate, score distributions and drift.

Three signals a claims AI programme has to watch:

* **Override rate** — how often humans contradict a substantive AI
  recommendation. Too high questions the model; near-zero may mean people are
  rubber-stamping rather than agreeing.
* **Score distributions over time** — whether the mix of claims the model sees
  is shifting.
* **Drift** — population stability index (PSI) between an earlier and a later
  period.

The drift signal here is a demonstration of the mechanism, not a reliable
measurement: ~1,000 claims over two months is far too little data, and the
comparison is between halves of the same period rather than against the
training distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from claimpilot import config


class SignalStatus(str, Enum):
    OK = "OK"
    WARN = "Warning"
    ALERT = "Alert"
    INSUFFICIENT = "Insufficient data"

    @property
    def icon(self) -> str:
        return {"OK": "🟢", "Warning": "🟠", "Alert": "🔴", "Insufficient data": "⚪"}[
            self.value
        ]


@dataclass
class MonitoringSignal:
    name: str
    value: float | None
    status: SignalStatus
    detail: str
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 4) if self.value is not None else None,
            "status": self.status.value,
            "threshold": self.threshold,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Override rate
# ---------------------------------------------------------------------------
def override_signal(log) -> MonitoringSignal:
    """Assess the override rate against both an upper and a lower bound.

    A near-zero override rate is not a success. If humans never disagree with
    the AI, either the model is perfect or nobody is really reviewing. The
    second is far more likely, so a floor is monitored as well as a ceiling.
    """
    cfg = config.assumption("monitoring")
    warn = float(cfg["override_rate_warn"])
    floor = float(cfg["override_rate_floor"])

    eligible = log.override_eligible_count
    if eligible < 20:
        return MonitoringSignal(
            name="Override rate",
            value=log.override_rate if eligible else None,
            status=SignalStatus.INSUFFICIENT,
            detail=(
                f"Only {eligible} decisions where the AI made a substantive "
                "recommendation. Too few to read anything into."
            ),
            threshold=warn,
        )

    rate = log.override_rate
    if rate > warn:
        status = SignalStatus.ALERT
        detail = (
            f"Humans contradicted the AI on {rate:.1%} of {eligible} eligible "
            f"decisions, above the {warn:.0%} threshold. Investigate whether "
            "the model or the routing rules are mis-calibrated."
        )
    elif rate < floor:
        status = SignalStatus.WARN
        detail = (
            f"Override rate of {rate:.1%} is below the {floor:.0%} floor. "
            "Near-total agreement may indicate rubber-stamping rather than "
            "genuine review."
        )
    else:
        status = SignalStatus.OK
        detail = (
            f"{rate:.1%} of {eligible} eligible decisions were overridden, "
            f"within the {floor:.0%}-{warn:.0%} range."
        )

    return MonitoringSignal("Override rate", rate, status, detail, warn)


# ---------------------------------------------------------------------------
# Drift (population stability index)
# ---------------------------------------------------------------------------
def population_stability_index(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    bins: int | None = None,
) -> float:
    """PSI between a baseline and a later distribution.

    Bins are cut on the baseline's quantiles. Empty bins are floored to a small
    value so the result stays finite rather than exploding to infinity.
    """
    bins = bins or int(config.assumption("monitoring", "psi_bins"))
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    expected_share = np.histogram(expected, bins=edges)[0] / len(expected)
    actual_share = np.histogram(actual, bins=edges)[0] / len(actual)

    floor = 1e-6
    expected_share = np.clip(expected_share, floor, None)
    actual_share = np.clip(actual_share, floor, None)

    return float(
        np.sum((actual_share - expected_share) * np.log(actual_share / expected_share))
    )


def drift_signal(
    scored: pd.DataFrame, claims: pd.DataFrame, column: str = "fraud_probability"
) -> MonitoringSignal:
    """Compare the later half of the claim period against the earlier half."""
    cfg = config.assumption("monitoring")
    warn = float(cfg["drift_psi_warn"])
    alert = float(cfg["drift_psi_alert"])

    frame = scored[["claim_id", column]].merge(
        claims[["claim_id", "incident_date"]], on="claim_id", how="left"
    )
    frame = frame.dropna(subset=["incident_date"]).sort_values("incident_date")

    if len(frame) < 100:
        return MonitoringSignal(
            "Score drift (PSI)", None, SignalStatus.INSUFFICIENT,
            "Fewer than 100 scored claims; drift cannot be assessed.", warn,
        )

    midpoint = len(frame) // 2
    psi = population_stability_index(
        frame[column].iloc[:midpoint], frame[column].iloc[midpoint:]
    )

    if np.isnan(psi):
        status, detail = SignalStatus.INSUFFICIENT, "PSI could not be computed."
    elif psi >= alert:
        status = SignalStatus.ALERT
        detail = f"PSI {psi:.3f} is at or above the {alert} alert level: the score distribution has shifted significantly."
    elif psi >= warn:
        status = SignalStatus.WARN
        detail = f"PSI {psi:.3f} is above the {warn} warning level: a moderate shift in the score distribution."
    else:
        status = SignalStatus.OK
        detail = f"PSI {psi:.3f} is below the {warn} warning level: no material shift between the two periods."

    detail += (
        " Note: this compares halves of the same two-month period, not live "
        "scores against the training distribution, and ~1,000 claims is far "
        "too few for a dependable signal."
    )
    return MonitoringSignal("Score drift (PSI)", psi, status, detail, warn)


# ---------------------------------------------------------------------------
# Distributions over time
# ---------------------------------------------------------------------------
def scores_over_time(
    scored: pd.DataFrame, claims: pd.DataFrame, freq: str = "W"
) -> pd.DataFrame:
    """Weekly mean fraud score and band mix, for the trend charts."""
    frame = scored.merge(
        claims[["claim_id", "incident_date"]], on="claim_id", how="left"
    ).dropna(subset=["incident_date"])

    frame["period"] = pd.to_datetime(frame["incident_date"]).dt.to_period(freq).dt.start_time

    grouped = frame.groupby("period").agg(
        claims=("claim_id", "count"),
        mean_fraud_probability=("fraud_probability", "mean"),
        stp_rate=("auto_approved", "mean"),
        siu_rate=("destination", lambda s: (s == "siu_queue").mean()),
        high_severity_rate=("severity_band", lambda s: (s == "High").mean()),
    )
    return grouped.reset_index()


def band_mix_over_time(
    scored: pd.DataFrame, claims: pd.DataFrame, band: str = "fraud_band", freq: str = "W"
) -> pd.DataFrame:
    frame = scored.merge(
        claims[["claim_id", "incident_date"]], on="claim_id", how="left"
    ).dropna(subset=["incident_date"])
    frame["period"] = pd.to_datetime(frame["incident_date"]).dt.to_period(freq).dt.start_time

    mix = (
        frame.groupby(["period", band]).size().rename("claims").reset_index()
    )
    totals = mix.groupby("period")["claims"].transform("sum")
    mix["share"] = mix["claims"] / totals
    return mix


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def all_signals(log, scored: pd.DataFrame, claims: pd.DataFrame) -> list[MonitoringSignal]:
    return [override_signal(log), drift_signal(scored, claims)]


def monitoring_summary(log, scored: pd.DataFrame, claims: pd.DataFrame) -> dict[str, Any]:
    signals = all_signals(log, scored, claims)
    return {
        "signals": [s.to_dict() for s in signals],
        "alerts": len([s for s in signals if s.status is SignalStatus.ALERT]),
        "warnings": len([s for s in signals if s.status is SignalStatus.WARN]),
        "drift_note": " ".join(str(config.assumption("monitoring", "drift_note")).split()),
    }
