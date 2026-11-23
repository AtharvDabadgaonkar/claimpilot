"""The three claims that carry the two-minute demo story.

Chosen by *rule*, not by claim id, so the Overview page cannot drift from the
data. If a threshold in ``config/assumptions.yaml`` changes and a different
claim now fits the story, the page follows it automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from claimpilot.routing import Destination


@dataclass(frozen=True)
class StoryClaim:
    """One beat of the demo narrative."""

    beat: str
    headline: str
    claim_id: str
    destination: str
    destination_label: str
    severity_band: str
    complexity_band: str
    fraud_band: str
    fraud_probability: float
    routing_rule: str
    routing_reason: str
    total_claim_amount: float
    requires_human: bool
    auto_approved: bool
    page: str


def _row_to_beat(
    row: pd.Series, beat: str, headline: str, page: str, amount: float
) -> StoryClaim:
    return StoryClaim(
        beat=beat,
        headline=headline,
        claim_id=str(row["claim_id"]),
        destination=str(row["destination"]),
        destination_label=str(row["destination_label"]),
        severity_band=str(row["severity_band"]),
        complexity_band=str(row["complexity_band"]),
        fraud_band=str(row["fraud_band"]),
        fraud_probability=float(row["fraud_probability"]),
        routing_rule=str(row["routing_rule"]),
        routing_reason=str(row["routing_reason"]),
        total_claim_amount=float(amount),
        requires_human=bool(row["requires_human"]),
        auto_approved=bool(row["auto_approved"]),
        page=page,
    )


def pick_story_claims(
    scored: pd.DataFrame, claims: pd.DataFrame
) -> list[StoryClaim]:
    """Pick one fast-tracked, one SIU-referred and one adjuster claim.

    Selection is deterministic: lowest fraud probability among fast-tracked
    claims, highest among SIU referrals, and the largest exposure among the
    claims a human adjuster is asked to decide. Ties break on claim id.
    """
    amounts = claims.set_index("claim_id")["total_claim_amount"]
    beats: list[StoryClaim] = []

    fast = scored[scored["destination"] == Destination.STP_AUTO_APPROVE.value]
    if not fast.empty:
        row = fast.sort_values(["fraud_probability", "claim_id"]).iloc[0]
        beats.append(
            _row_to_beat(
                row,
                "1 · Simple claim, fast-tracked",
                "Low severity, low complexity, low fraud signal, coverage "
                "confirmed — so it clears without a human touching it.",
                "Triage & Routing",
                amounts.get(row["claim_id"], 0.0),
            )
        )

    siu = scored[scored["destination"] == Destination.SIU_QUEUE.value]
    if not siu.empty:
        row = siu.sort_values(
            ["fraud_probability", "claim_id"], ascending=[False, True]
        ).iloc[0]
        beats.append(
            _row_to_beat(
                row,
                "2 · Suspicious claim, referred to SIU",
                "A high fraud signal with named red flags. Referred for "
                "investigation — never denied by the system.",
                "SIU Queue",
                amounts.get(row["claim_id"], 0.0),
            )
        )

    human = scored[
        scored["destination"].isin(
            [Destination.ADJUSTER_QUEUE.value, Destination.SENIOR_ADJUSTER.value]
        )
    ]
    if not human.empty:
        merged = human.merge(
            claims[["claim_id", "total_claim_amount"]], on="claim_id", how="left"
        )
        row = merged.sort_values(
            ["total_claim_amount", "claim_id"], ascending=[False, True]
        ).iloc[0]
        beats.append(
            _row_to_beat(
                row,
                "3 · Everything else, decided by a person",
                "The copilot drafts a summary, suggests a reserve from the "
                "model and lists the missing documents. The adjuster decides.",
                "Adjuster Workbench",
                float(row["total_claim_amount"]),
            )
        )

    return beats


def story_frame(beats: list[StoryClaim]) -> pd.DataFrame:
    """The beats as a display table for the Overview page."""
    return pd.DataFrame(
        [
            {
                "Beat": b.beat,
                "Claim": b.claim_id,
                "Amount": b.total_claim_amount,
                "Severity": b.severity_band,
                "Complexity": b.complexity_band,
                "Fraud": f"{b.fraud_band} ({b.fraud_probability:.0%})",
                "Routed to": b.destination_label,
                "Rule fired": b.routing_rule,
                "Human decides": "No" if b.auto_approved else "Yes",
                "Where to look": b.page,
            }
            for b in beats
        ]
    )
