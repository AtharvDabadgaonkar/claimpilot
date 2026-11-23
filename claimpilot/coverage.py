"""Coverage check against the mock policy.

Answers one question before triage runs: on the date of loss, did this policy
cover this kind of loss, and what are the deductible and limits?

A coverage *question* is never a denial. This module returns a status and the
reasons behind it; deciding what to do about a failed check is a human's job
(see the no-auto-deny rule in claimpilot.routing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any

import pandas as pd


class CoverageStatus(str, Enum):
    CONFIRMED = "Confirmed"
    QUESTION = "Coverage question"
    NOT_IN_FORCE = "Not in force at date of loss"
    UNKNOWN = "Unknown"


# Loss types that need physical-damage cover. A liability-only policy raises a
# coverage question for these, because liability covers damage the insured does
# to others, not damage to the insured's own vehicle.
_PHYSICAL_DAMAGE_LOSSES = {
    "Single Vehicle Collision",
    "Multi-vehicle Collision",
    "Parked Car",
}
# Theft is covered by comprehensive, which liability-only also lacks.
_COMPREHENSIVE_LOSSES = {"Vehicle Theft"}


@dataclass
class CoverageResult:
    """Outcome of the coverage check, with the reasons that produced it."""

    status: CoverageStatus
    reasons: list[str] = field(default_factory=list)
    deductible_usd: float | None = None
    limit_per_person_usd: float | None = None
    limit_per_occurrence_usd: float | None = None
    policy_term_start: date | None = None
    policy_term_end: date | None = None
    days_outside_term: int | None = None

    @property
    def is_confirmed(self) -> bool:
        return self.status is CoverageStatus.CONFIRMED

    @property
    def blocks_fast_track(self) -> bool:
        """Anything other than confirmed cover must be seen by a human."""
        return self.status is not CoverageStatus.CONFIRMED

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage_status": self.status.value,
            "coverage_reasons": list(self.reasons),
            "deductible_usd": self.deductible_usd,
            "limit_per_person_usd": self.limit_per_person_usd,
            "limit_per_occurrence_usd": self.limit_per_occurrence_usd,
            "days_outside_term": self.days_outside_term,
        }


def _as_date(value: Any) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if ts is pd.NaT or pd.isna(ts):
        return None
    return ts.date()


def check_coverage(
    policy: dict[str, Any] | pd.Series,
    loss_date: Any,
    loss_type: str,
    claim_amount_usd: float | None = None,
) -> CoverageResult:
    """Check a loss against a mock policy.

    Returns ``NOT_IN_FORCE`` when the loss falls outside the policy term,
    ``QUESTION`` when cover is in force but the loss type may not be covered
    or the amount strains the limits, and ``CONFIRMED`` otherwise.
    """
    policy = dict(policy)
    reasons: list[str] = []

    term_start = _as_date(policy.get("term_start"))
    term_end = _as_date(policy.get("term_end"))
    loss = _as_date(loss_date)

    deductible = policy.get("deductible_usd")
    per_person = policy.get("csl_per_person_usd")
    per_occurrence = policy.get("csl_per_occurrence_usd")
    coverage_type = str(policy.get("coverage_type", ""))

    result = CoverageResult(
        status=CoverageStatus.UNKNOWN,
        deductible_usd=float(deductible) if deductible is not None else None,
        limit_per_person_usd=float(per_person) if per_person is not None else None,
        limit_per_occurrence_usd=(
            float(per_occurrence) if per_occurrence is not None else None
        ),
        policy_term_start=term_start,
        policy_term_end=term_end,
    )

    if loss is None or term_start is None or term_end is None:
        result.status = CoverageStatus.UNKNOWN
        result.reasons = ["Policy term or date of loss is missing."]
        return result

    # 1. In force at the date of loss?
    if loss < term_start:
        result.status = CoverageStatus.NOT_IN_FORCE
        result.days_outside_term = (term_start - loss).days
        result.reasons = [
            f"Loss on {loss} predates policy inception on {term_start} "
            f"by {result.days_outside_term} days."
        ]
        return result
    if loss > term_end:
        result.status = CoverageStatus.NOT_IN_FORCE
        result.days_outside_term = (loss - term_end).days
        result.reasons = [
            f"Policy expired on {term_end}; loss occurred {result.days_outside_term} "
            "days later."
        ]
        return result

    reasons.append(f"Policy in force on {loss} (term {term_start} to {term_end}).")

    # 2. Does the coverage type reach this loss?
    liability_only = "liability only" in coverage_type.lower()
    if liability_only and loss_type in _PHYSICAL_DAMAGE_LOSSES:
        reasons.append(
            f"Policy is {coverage_type}; damage to the insured vehicle from a "
            f"{loss_type.lower()} is not covered by liability cover."
        )
        result.status = CoverageStatus.QUESTION
        result.reasons = reasons
        return result
    if liability_only and loss_type in _COMPREHENSIVE_LOSSES:
        reasons.append(
            f"Policy is {coverage_type}; {loss_type.lower()} requires "
            "comprehensive cover."
        )
        result.status = CoverageStatus.QUESTION
        result.reasons = reasons
        return result

    reasons.append(f"{coverage_type} covers a {loss_type.lower()}.")

    # 3. Does the claimed amount strain the limits?
    if (
        claim_amount_usd is not None
        and per_occurrence
        and claim_amount_usd > float(per_occurrence)
    ):
        reasons.append(
            f"Claimed ${claim_amount_usd:,.0f} exceeds the per-occurrence "
            f"limit of ${float(per_occurrence):,.0f}; excess is uninsured."
        )
        result.status = CoverageStatus.QUESTION
        result.reasons = reasons
        return result

    if deductible:
        reasons.append(f"Deductible of ${float(deductible):,.0f} applies.")

    result.status = CoverageStatus.CONFIRMED
    result.reasons = reasons
    return result


def check_coverage_batch(
    claims: pd.DataFrame, policies: pd.DataFrame
) -> pd.DataFrame:
    """Run the coverage check across every claim. Used by triage and the backtest."""
    merged = claims[["claim_id", "incident_date", "incident_type", "total_claim_amount"]].merge(
        policies, on="claim_id", how="left"
    )
    rows = []
    for _, row in merged.iterrows():
        result = check_coverage(
            row,
            loss_date=row["incident_date"],
            loss_type=row["incident_type"],
            claim_amount_usd=float(row["total_claim_amount"]),
        )
        rows.append({"claim_id": row["claim_id"], **result.to_dict()})
    return pd.DataFrame(rows)
