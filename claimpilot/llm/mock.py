"""Rule-based mock provider — the default, and the only one used in tests.

This is deliberately NOT a language model. It is a deterministic template
engine over the claim's own structured fields, which gives the demo three
properties a real LLM would not:

* it cannot invent a dollar amount that is not in the claim record;
* it produces the same output for the same claim every time;
* it needs no API key, so the deployed demo always works.

The UI labels this mode "Demo mode: rule-based AI".
"""

from __future__ import annotations

import re
from typing import Any

from claimpilot.llm.base import LLMProvider


class MockProvider(LLMProvider):
    """Deterministic, template-driven stand-in for a hosted LLM."""

    name = "mock-rule-based"
    model = "claimpilot-rules-v0.1"
    is_mock = True

    def _generate(self, prompt: str, **kwargs: Any) -> str:
        task = kwargs.get("task", self._infer_task(prompt))
        handler = {
            "summary": self._summary,
            "next_action": self._next_action,
            "extract": self._extract_echo,
        }.get(task, self._summary)
        return handler(prompt, kwargs.get("fields", {}))

    @staticmethod
    def _infer_task(prompt: str) -> str:
        low = prompt.lower()
        if "next best action" in low or "next action" in low:
            return "next_action"
        if "extract" in low:
            return "extract"
        return "summary"

    # -- Task handlers -------------------------------------------------------

    def _summary(self, prompt: str, fields: dict[str, Any]) -> str:
        """Summarise strictly from supplied structured fields."""
        if not fields:
            # Fall back to echoing the loss description sentences we were given.
            sentences = re.split(r"(?<=[.!?])\s+", prompt.strip())
            body = " ".join(s for s in sentences if s)[:400]
            return body or "No claim detail supplied."

        parts: list[str] = []
        itype = fields.get("incident_type")
        sev = fields.get("incident_severity")
        if itype:
            parts.append(f"{itype.lower()} reported as {str(sev).lower()}")
        if fields.get("has_injury"):
            n = fields.get("bodily_injuries", 0)
            parts.append(f"{n} bodily injury claim(s) reported")
        else:
            parts.append("no bodily injury reported")
        if fields.get("multi_party"):
            parts.append(f"{fields.get('number_of_vehicles_involved')} vehicles involved")
        if fields.get("missing_police_report"):
            parts.append("police report not on file")
        amount = fields.get("total_claim_amount")
        if amount is not None:
            parts.append(f"total claimed ${float(amount):,.0f}")

        lead = "; ".join(parts)
        return f"{lead.capitalize()}."

    def _next_action(self, prompt: str, fields: dict[str, Any]) -> str:
        """Suggest the next step from documented gaps only."""
        actions: list[str] = []
        if fields.get("missing_police_report"):
            actions.append("request the police report")
        if fields.get("missing_property_damage_info"):
            actions.append("confirm whether there was property damage")
        if fields.get("has_injury"):
            actions.append("request medical records and set an injury reserve")
        if fields.get("multi_party"):
            actions.append("identify all parties and check subrogation potential")
        if not actions:
            actions.append("verify coverage and proceed to settlement review")
        return "Next: " + "; then ".join(actions) + "."

    def _extract_echo(self, prompt: str, fields: dict[str, Any]) -> str:
        return (
            "Extraction is handled by the rule-based extractor in "
            "claimpilot.intake, not by this provider."
        )
