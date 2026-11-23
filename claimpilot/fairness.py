"""Fairness testing: are outcomes evenly distributed across groups?

Protected attributes are never model inputs (see
``protected_and_excluded_fields`` in the config), but that is not sufficient.
Other features can correlate with a protected characteristic, so the outcome
has to be measured, not assumed. This module measures two outcomes — the STP
fast-track rate and the SIU referral rate — across region, age band, vehicle
category and sex.

The headline statistic is the **disparate impact ratio**: the lowest group's
rate divided by the highest group's. A ratio below 0.8 is flagged.

On that 0.8: it is the EEOC's four-fifths rule from US *employment* law, used
here because it is widely understood. It is not an insurance standard and not
a legal test for claims handling. Insurance discrimination is governed by
state unfair trade practice and unfair claims settlement acts, which define no
single numeric threshold. A real carrier would agree its metrics with counsel
and its regulators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from claimpilot import config


@dataclass
class GroupRate:
    """One group's outcome rate."""

    group: str
    n: int
    favourable: int
    rate: float
    too_small: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "n": self.n,
            "favourable": self.favourable,
            "rate": round(self.rate, 4),
            "too_small": self.too_small,
        }


@dataclass
class FairnessResult:
    """Disparate impact for one outcome across one attribute."""

    attribute: str
    outcome: str
    groups: list[GroupRate]
    disparate_impact_ratio: float | None
    threshold: float
    lowest_group: str | None = None
    highest_group: str | None = None
    excluded_small_groups: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        """Below the threshold, and therefore warranting investigation."""
        if self.disparate_impact_ratio is None:
            return False
        return self.disparate_impact_ratio < self.threshold

    @property
    def verdict(self) -> str:
        if self.disparate_impact_ratio is None:
            return "Not assessable"
        return "Flagged for review" if self.flagged else "Within threshold"

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute": self.attribute,
            "outcome": self.outcome,
            "disparate_impact_ratio": (
                round(self.disparate_impact_ratio, 4)
                if self.disparate_impact_ratio is not None
                else None
            ),
            "threshold": self.threshold,
            "flagged": self.flagged,
            "verdict": self.verdict,
            "lowest_group": self.lowest_group,
            "highest_group": self.highest_group,
            "excluded_small_groups": list(self.excluded_small_groups),
            "groups": [g.to_dict() for g in self.groups],
        }


def disparate_impact_ratio(rates: dict[str, float]) -> float | None:
    """Lowest group rate divided by highest group rate.

    Returns ``None`` when there is nothing to compare (fewer than two groups)
    or when the highest rate is zero, in which case no group received the
    outcome at all and a ratio would be meaningless rather than perfect.
    """
    if len(rates) < 2:
        return None
    highest = max(rates.values())
    if highest <= 0:
        return None
    return min(rates.values()) / highest


def assess(
    frame: pd.DataFrame,
    attribute: str,
    outcome_column: str,
    outcome_label: str | None = None,
    min_group_size: int | None = None,
    threshold: float | None = None,
) -> FairnessResult:
    """Measure one boolean outcome across the levels of one attribute."""
    threshold = (
        threshold
        if threshold is not None
        else float(config.assumption("fairness", "disparate_impact_threshold"))
    )
    min_group_size = (
        min_group_size
        if min_group_size is not None
        else int(config.assumption("fairness", "min_group_size"))
    )

    groups: list[GroupRate] = []
    comparable: dict[str, float] = {}
    excluded: list[str] = []

    for value, subset in frame.groupby(attribute, dropna=False):
        n = int(len(subset))
        favourable = int(subset[outcome_column].sum())
        rate = favourable / n if n else 0.0
        too_small = n < min_group_size
        groups.append(
            GroupRate(
                group=str(value),
                n=n,
                favourable=favourable,
                rate=rate,
                too_small=too_small,
            )
        )
        # A group of 6 produces a wild rate that would dominate the ratio.
        # Report it, but keep it out of the comparison.
        if too_small:
            excluded.append(str(value))
        else:
            comparable[str(value)] = rate

    groups.sort(key=lambda g: g.rate)
    ratio = disparate_impact_ratio(comparable)

    lowest = highest = None
    if comparable:
        lowest = min(comparable, key=lambda k: comparable[k])
        highest = max(comparable, key=lambda k: comparable[k])

    return FairnessResult(
        attribute=attribute,
        outcome=outcome_label or outcome_column,
        groups=groups,
        disparate_impact_ratio=ratio,
        threshold=threshold,
        lowest_group=lowest,
        highest_group=highest,
        excluded_small_groups=excluded,
    )


def build_fairness_frame(
    scored: pd.DataFrame, claims: pd.DataFrame
) -> pd.DataFrame:
    """Join routing outcomes to the protected attributes used for testing."""
    attributes = list(config.assumption("fairness", "test_groups").keys())
    available = [a for a in attributes if a in claims.columns]
    frame = scored[["claim_id", "destination", "auto_approved"]].merge(
        claims[["claim_id", *available]], on="claim_id", how="left"
    )
    frame["stp"] = frame["auto_approved"].astype(bool)
    frame["siu"] = frame["destination"].eq("siu_queue")
    return frame


def assess_all(
    scored: pd.DataFrame, claims: pd.DataFrame
) -> list[FairnessResult]:
    """Every tested outcome across every tested attribute."""
    frame = build_fairness_frame(scored, claims)
    attributes = [
        a
        for a in config.assumption("fairness", "test_groups")
        if a in frame.columns
    ]

    results: list[FairnessResult] = []
    for attribute in attributes:
        results.append(
            assess(frame, attribute, "stp", outcome_label="STP fast-track rate")
        )
        results.append(
            assess(frame, attribute, "siu", outcome_label="SIU referral rate")
        )
    return results


def summary(results: list[FairnessResult]) -> dict[str, Any]:
    """Portfolio-level fairness summary for metrics.json."""
    assessable = [r for r in results if r.disparate_impact_ratio is not None]
    flagged = [r for r in assessable if r.flagged]
    return {
        "threshold": float(config.assumption("fairness", "disparate_impact_threshold")),
        "tests_run": len(results),
        "tests_assessable": len(assessable),
        "tests_flagged": len(flagged),
        "flagged": [
            {
                "attribute": r.attribute,
                "outcome": r.outcome,
                "ratio": round(r.disparate_impact_ratio or 0, 4),
                "lowest_group": r.lowest_group,
                "highest_group": r.highest_group,
            }
            for r in flagged
        ],
        "worst_ratio": (
            round(min(r.disparate_impact_ratio for r in assessable), 4)
            if assessable
            else None
        ),
        "results": [r.to_dict() for r in results],
        "threshold_note": " ".join(
            str(config.assumption("fairness", "threshold_note")).split()
        ),
    }
