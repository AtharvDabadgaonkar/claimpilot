"""Triage scoring: severity, complexity and fraud signal, each with reasons.

Every score carries human-readable **reason codes**. A score without reasons is
not usable in claims: the adjuster has to be able to disagree with it, and a
regulator has to be able to follow it. The bands and weights live in
``config/assumptions.yaml`` so they can be reviewed without reading code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from claimpilot import config
from claimpilot.coverage import CoverageStatus

TRIAGE_VERSION = "triage-rules-v0.1"


class Band(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

    @property
    def rank(self) -> int:
        return {"Low": 0, "Medium": 1, "High": 2}[self.value]

    @classmethod
    def from_rank(cls, rank: int) -> Band:
        return [cls.LOW, cls.MEDIUM, cls.HIGH][max(0, min(2, rank))]


@dataclass
class Reason:
    """One explainable contribution to a score."""

    code: str
    text: str
    points: float = 0.0

    def __str__(self) -> str:
        return self.text


@dataclass
class Score:
    band: Band
    value: float
    reasons: list[Reason] = field(default_factory=list)

    @property
    def reason_codes(self) -> list[str]:
        return [r.code for r in self.reasons]

    @property
    def reason_texts(self) -> list[str]:
        return [r.text for r in self.reasons]


@dataclass
class TriageResult:
    claim_id: str
    severity: Score
    complexity: Score
    fraud: Score
    coverage_status: CoverageStatus
    triage_version: str = TRIAGE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "severity_band": self.severity.band.value,
            "severity_value": self.severity.value,
            "severity_reasons": self.severity.reason_texts,
            "severity_codes": self.severity.reason_codes,
            "complexity_band": self.complexity.band.value,
            "complexity_points": self.complexity.value,
            "complexity_reasons": self.complexity.reason_texts,
            "complexity_codes": self.complexity.reason_codes,
            "fraud_band": self.fraud.band.value,
            "fraud_probability": self.fraud.value,
            "fraud_reasons": self.fraud.reason_texts,
            "fraud_codes": self.fraud.reason_codes,
            "coverage_status": self.coverage_status.value,
            "triage_version": self.triage_version,
        }


# ---------------------------------------------------------------------------
# Severity — expected cost band
# ---------------------------------------------------------------------------
def score_severity(claim: dict[str, Any]) -> Score:
    cfg = config.assumption("triage", "severity")
    low_max = float(cfg["low_max_usd"])
    med_max = float(cfg["medium_max_usd"])

    amount = float(claim.get("total_claim_amount") or 0.0)
    reasons: list[Reason] = []

    if amount <= low_max:
        band = Band.LOW
    elif amount <= med_max:
        band = Band.MEDIUM
    else:
        band = Band.HIGH

    reasons.append(
        Reason(
            "SEV_AMOUNT",
            f"Claimed ${amount:,.0f} falls in the {band.value.lower()} cost band "
            f"(low ≤ ${low_max:,.0f}, medium ≤ ${med_max:,.0f}).",
        )
    )

    if cfg.get("total_loss_forces_high") and str(
        claim.get("incident_severity", "")
    ) == "Total Loss":
        if band is not Band.HIGH:
            reasons.append(
                Reason(
                    "SEV_TOTAL_LOSS",
                    "Reported as a total loss, which forces the high severity band.",
                )
            )
            band = Band.HIGH
        else:
            reasons.append(
                Reason("SEV_TOTAL_LOSS", "Reported as a total loss.")
            )

    if claim.get("has_injury"):
        reasons.append(
            Reason(
                "SEV_INJURY_EXPOSURE",
                f"{int(claim.get('bodily_injuries') or 0)} bodily injury claim(s) "
                "add indemnity exposure beyond vehicle damage.",
            )
        )

    return Score(band=band, value=amount, reasons=reasons)


# ---------------------------------------------------------------------------
# Complexity — how much human work this claim needs
# ---------------------------------------------------------------------------
def score_complexity(
    claim: dict[str, Any], coverage_status: CoverageStatus
) -> Score:
    cfg = config.assumption("triage", "complexity")
    pts = cfg["points"]
    sev_cfg = config.assumption("triage", "severity")

    reasons: list[Reason] = []
    total = 0.0

    def add(code: str, text: str, points: float) -> None:
        nonlocal total
        total += points
        reasons.append(Reason(code, text, points))

    injuries = int(claim.get("bodily_injuries") or 0)
    if claim.get("has_injury"):
        add("CPX_INJURY", f"{injuries} reported bodily injury claim(s).", pts["injury_reported"])
        if injuries > 1:
            add(
                "CPX_MULTI_INJURY",
                "More than one injured party increases negotiation and medical review.",
                pts["multiple_injuries"],
            )

    vehicles = int(claim.get("number_of_vehicles_involved") or 1)
    if claim.get("multi_party"):
        add("CPX_MULTI_PARTY", f"{vehicles} vehicles involved; liability must be apportioned.", pts["multi_party"])
    if vehicles >= 3:
        add("CPX_MANY_VEHICLES", f"{vehicles} vehicles involved.", pts["many_vehicles"])

    if claim.get("missing_police_report"):
        add("CPX_NO_POLICE_REPORT", "Police report not on file.", pts["missing_police_report"])
    if claim.get("missing_property_damage_info"):
        add(
            "CPX_NO_PROPERTY_INFO",
            "Property damage not yet confirmed.",
            pts["missing_property_damage_info"],
        )

    if coverage_status is CoverageStatus.NOT_IN_FORCE:
        add(
            "CPX_COVERAGE_NOT_IN_FORCE",
            "Policy was not in force at the date of loss.",
            pts["coverage_not_in_force"],
        )
    elif coverage_status in (CoverageStatus.QUESTION, CoverageStatus.UNKNOWN):
        add("CPX_COVERAGE_QUESTION", "Open coverage question on this claim.", pts["coverage_question"])

    witnesses = int(claim.get("witnesses") or 0)
    if witnesses == 0 and claim.get("multi_party"):
        add(
            "CPX_NO_WITNESS_DISPUTE",
            "Multi-party loss with no witnesses; liability may be disputed.",
            pts["no_witnesses_disputed"],
        )

    amount = float(claim.get("total_claim_amount") or 0.0)
    if amount > float(sev_cfg["medium_max_usd"]):
        add("CPX_HIGH_VALUE", f"High claimed value (${amount:,.0f}).", pts["high_value"])

    if not reasons:
        reasons.append(
            Reason("CPX_NONE", "No complexity drivers identified on this claim.")
        )

    if total <= float(cfg["low_max_points"]):
        band = Band.LOW
    elif total <= float(cfg["medium_max_points"]):
        band = Band.MEDIUM
    else:
        band = Band.HIGH

    return Score(band=band, value=total, reasons=reasons)


# ---------------------------------------------------------------------------
# Fraud signal — model probability plus legible red flags
# ---------------------------------------------------------------------------
def red_flags(claim: dict[str, Any]) -> tuple[float, list[Reason]]:
    """Classic P&C indicators, kept few and explainable."""
    cfg = config.assumption("triage", "fraud", "red_flags")
    reasons: list[Reason] = []
    points = 0.0

    hour = claim.get("incident_hour_of_the_day")
    if hour is not None and int(hour) in set(cfg["late_night_incident"]["hours"]):
        p = float(cfg["late_night_incident"]["points"])
        points += p
        reasons.append(
            Reason("FRD_LATE_NIGHT", f"Loss occurred at {int(hour)}:00, outside normal hours.", p)
        )

    amount = float(claim.get("total_claim_amount") or 0.0)
    rule = cfg["no_police_report_high_value"]
    if claim.get("missing_police_report") and amount >= float(rule["min_amount_usd"]):
        p = float(rule["points"])
        points += p
        reasons.append(
            Reason(
                "FRD_NO_REPORT_HIGH_VALUE",
                f"No police report on a ${amount:,.0f} claim.",
                p,
            )
        )

    tenure = claim.get("months_as_customer")
    rule = cfg["recent_policy_short_tenure"]
    if tenure is not None and float(tenure) <= float(rule["max_months_as_customer"]):
        p = float(rule["points"])
        points += p
        reasons.append(
            Reason(
                "FRD_SHORT_TENURE",
                f"Policy only {int(tenure)} months old at the date of loss.",
                p,
            )
        )

    witnesses = int(claim.get("witnesses") or 0)
    if witnesses == 0 and claim.get("missing_police_report"):
        p = float(cfg["no_witnesses_no_report"]["points"])
        points += p
        reasons.append(
            Reason(
                "FRD_NO_CORROBORATION",
                "No witnesses and no police report; nothing corroborates the account.",
                p,
            )
        )

    premium = float(claim.get("policy_annual_premium") or 0.0)
    rule = cfg["high_claim_low_premium"]
    if premium > 0 and amount / premium >= float(rule["ratio"]):
        p = float(rule["points"])
        points += p
        reasons.append(
            Reason(
                "FRD_CLAIM_PREMIUM_RATIO",
                f"Claim is {amount / premium:.0f}x the annual premium.",
                p,
            )
        )

    return points, reasons


def score_fraud(claim: dict[str, Any], probability: float) -> Score:
    cfg = config.assumption("triage", "fraud")
    reasons: list[Reason] = [
        Reason(
            "FRD_MODEL_SCORE",
            f"Model fraud probability {probability:.0%} "
            f"(bands: low \u2264 {float(cfg['low_max']):.0%}, "
            f"medium \u2264 {float(cfg['medium_max']):.0%}).",
        )
    ]

    if probability <= float(cfg["low_max"]):
        band = Band.LOW
    elif probability <= float(cfg["medium_max"]):
        band = Band.MEDIUM
    else:
        band = Band.HIGH

    flag_points, flag_reasons = red_flags(claim)
    reasons.extend(flag_reasons)

    if flag_points >= float(cfg["escalate_band_at_points"]) and band is not Band.HIGH:
        escalated = Band.from_rank(band.rank + 1)
        reasons.append(
            Reason(
                "FRD_FLAG_ESCALATION",
                f"{int(flag_points)} red-flag points escalate the fraud band from "
                f"{band.value} to {escalated.value}.",
                flag_points,
            )
        )
        band = escalated

    return Score(band=band, value=round(float(probability), 4), reasons=reasons)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def triage_claim(
    claim: dict[str, Any],
    fraud_probability: float,
    coverage_status: CoverageStatus,
) -> TriageResult:
    return TriageResult(
        claim_id=str(claim.get("claim_id", "")),
        severity=score_severity(claim),
        complexity=score_complexity(claim, coverage_status),
        fraud=score_fraud(claim, fraud_probability),
        coverage_status=coverage_status,
    )


def triage_batch(
    claims: pd.DataFrame,
    coverage: pd.DataFrame,
    probabilities: np.ndarray | None = None,
) -> pd.DataFrame:
    """Triage every claim. Used by the dashboard and the backtest."""
    from claimpilot.fraud_model import predict_proba

    if probabilities is None:
        probabilities = predict_proba(claims)

    status_by_id = dict(
        zip(coverage["claim_id"], coverage["coverage_status"], strict=True)
    )

    rows = []
    for i, (_, claim) in enumerate(claims.iterrows()):
        status = CoverageStatus(status_by_id.get(claim["claim_id"], "Unknown"))
        result = triage_claim(claim.to_dict(), float(probabilities[i]), status)
        rows.append(result.to_dict())
    return pd.DataFrame(rows)
