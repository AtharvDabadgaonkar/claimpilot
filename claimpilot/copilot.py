"""Adjuster copilot: summary, next-best action, missing documents, reserve.

Two constraints shape this module, and both are tested:

1. **The copilot only uses the claim's own data.** It is handed a specific set
   of fields and has no access to anything else.
2. **It never invents an amount.** The reserve comes from
   :mod:`claimpilot.reserve`; the language model may phrase it but never
   originates it. Every figure the copilot states is traceable to either the
   claim record or the reserve model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from claimpilot.llm import LLMProvider, get_provider
from claimpilot.reserve import ReserveSuggestion, suggest_reserve

# The only claim fields the copilot is allowed to read. Anything else is not
# passed in, so the summary cannot drift onto the policyholder's demographics.
ALLOWED_FIELDS = (
    "claim_id",
    "incident_type",
    "incident_severity",
    "collision_type",
    "total_claim_amount",
    "bodily_injuries",
    "has_injury",
    "multi_party",
    "number_of_vehicles_involved",
    "witnesses",
    "missing_police_report",
    "missing_property_damage_info",
    "incident_date",
    "severity_band",
    "complexity_band",
    "fraud_band",
    "fraud_probability",
    "coverage_status",
    "destination",
    "destination_label",
)

# Documents a claim needs before it can be decided, and what triggers each.
_DOCUMENT_RULES: list[tuple[str, str, str]] = [
    (
        "Police report",
        "missing_police_report",
        "Not on file. Needed to establish the facts of loss and support subrogation.",
    ),
    (
        "Property damage confirmation",
        "missing_property_damage_info",
        "Not yet confirmed whether third-party property was damaged.",
    ),
]


@dataclass
class CopilotBriefing:
    """Everything the copilot offers on one claim."""

    claim_id: str
    summary: str
    next_best_action: str
    missing_documents: list[dict[str, str]]
    reserve: ReserveSuggestion
    triage_reasons: dict[str, list[str]] = field(default_factory=dict)
    provider_name: str = ""
    is_mock: bool = True
    pii_items_masked: int = 0

    @property
    def has_open_documents(self) -> bool:
        return bool(self.missing_documents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "summary": self.summary,
            "next_best_action": self.next_best_action,
            "missing_documents": [d["document"] for d in self.missing_documents],
            **self.reserve.to_dict(),
            "copilot_provider": self.provider_name,
            "copilot_is_mock": self.is_mock,
        }


def allowed_fields(claim: dict[str, Any]) -> dict[str, Any]:
    """Narrow a claim record down to what the copilot may see."""
    return {k: v for k, v in dict(claim).items() if k in ALLOWED_FIELDS}


def missing_documents(claim: dict[str, Any]) -> list[dict[str, str]]:
    """Checklist of what still has to be obtained."""
    items: list[dict[str, str]] = []
    for document, flag, why in _DOCUMENT_RULES:
        if claim.get(flag):
            items.append({"document": document, "why": why})

    if claim.get("has_injury"):
        items.append(
            {
                "document": "Medical records",
                "why": (
                    f"{int(claim.get('bodily_injuries') or 0)} reported bodily "
                    "injury claim(s); records needed to value the injury reserve."
                ),
            }
        )
    if claim.get("coverage_status") not in (None, "Confirmed"):
        items.append(
            {
                "document": "Coverage confirmation",
                "why": f"Coverage status is '{claim.get('coverage_status')}'.",
            }
        )
    if float(claim.get("total_claim_amount") or 0) > 0 and claim.get(
        "severity_band"
    ) in ("Medium", "High"):
        items.append(
            {
                "document": "Independent repair estimate",
                "why": "Claim value warrants an independent estimate before settlement.",
            }
        )
    return items


def next_best_action(
    claim: dict[str, Any], documents: list[dict[str, str]]
) -> str:
    """The single most useful next step, derived from documented gaps only."""
    if claim.get("coverage_status") == "Not in force at date of loss":
        return (
            "Confirm the policy term with underwriting before contacting the "
            "insured. If cover truly was not in force this still requires a "
            "human decision — the system cannot deny a claim."
        )
    if claim.get("fraud_band") == "High":
        return (
            "Refer to SIU before any payment. Preserve the file and do not "
            "signal the investigation to the insured."
        )
    if claim.get("coverage_status") == "Coverage question":
        return "Resolve the open coverage question before valuing the claim."
    if documents:
        first = documents[0]
        return f"Request the {first['document'].lower()} — {first['why']}"
    if claim.get("has_injury"):
        return "Set the injury reserve and schedule a medical review."
    return "Verify the repair estimate and proceed to settlement within authority."


def build_briefing(
    claim: dict[str, Any] | pd.Series,
    provider: LLMProvider | None = None,
    reserve_bundle: dict[str, Any] | None = None,
    narrative: str | None = None,
    triage_reasons: dict[str, list[str]] | None = None,
) -> CopilotBriefing:
    """Assemble the copilot's view of one claim.

    ``narrative`` is the claimant's free text. It is passed as ``context`` so
    the provider masks it before use; the copilot never reads it directly.
    """
    provider = provider or get_provider()
    fields = allowed_fields(claim)

    documents = missing_documents(fields)

    # The reserve is a MODEL output. The LLM is never asked for a number.
    reserve = suggest_reserve(claim, bundle=reserve_bundle)

    response = provider.complete(
        "Summarise this auto claim for the handling adjuster in one sentence.",
        context=narrative or "",
        task="summary",
        fields=fields,
    )

    return CopilotBriefing(
        claim_id=str(fields.get("claim_id", "")),
        summary=response.text,
        next_best_action=next_best_action(fields, documents),
        missing_documents=documents,
        reserve=reserve,
        triage_reasons=triage_reasons or {},
        provider_name=response.provider,
        is_mock=response.is_mock,
        pii_items_masked=response.pii_items_masked,
    )
