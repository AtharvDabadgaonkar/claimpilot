"""The app must work with no API key, and must never leak PII to a provider."""

from __future__ import annotations

import pytest

from claimpilot.llm import MockProvider, get_provider, provider_badge
from claimpilot.llm.base import LLMProvider


def test_defaults_to_mock_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = get_provider()
    assert provider.is_mock
    assert isinstance(provider, MockProvider)


def test_badge_announces_demo_mode():
    assert provider_badge(MockProvider()) == "Demo mode: rule-based AI"


def test_complete_masks_the_context_before_generating():
    provider = MockProvider()
    response = provider.complete(
        "Summarise this claim.",
        context="My name is Jane Doe, jane.doe@example.com, (555) 213-4455.",
    )
    assert response.pii_items_masked >= 3
    assert "jane.doe@example.com" not in response.text
    assert response.is_mock


def test_masking_cannot_be_bypassed_by_a_subclass():
    """complete() masks before _generate() is ever reached."""
    seen: list[str] = []

    class Spy(LLMProvider):
        name, model, is_mock = "spy", "spy-1", True

        def _generate(self, prompt: str, **kwargs) -> str:
            seen.append(prompt)
            return "ok"

    Spy().complete("Summarise.", context="Email jane.doe@example.com now.")
    assert seen and "jane.doe@example.com" not in seen[0]
    assert "[EMAIL_1]" in seen[0]


def test_summary_uses_only_supplied_fields(claims):
    """The copilot must not invent an amount."""
    row = claims.iloc[0].to_dict()
    response = MockProvider().complete(
        "Summarise this auto claim.", task="summary", fields=row
    )
    assert f"{row['total_claim_amount']:,.0f}" in response.text
    assert row["incident_type"].lower() in response.text.lower()


def test_summary_without_fields_invents_nothing():
    response = MockProvider().complete("Summarise this auto claim.", task="summary")
    assert "$" not in response.text


def test_next_action_is_driven_by_documented_gaps():
    response = MockProvider().complete(
        "What is the next best action?",
        task="next_action",
        fields={"missing_police_report": True, "has_injury": False},
    )
    assert "police report" in response.text.lower()


def test_mock_is_deterministic(claims):
    row = claims.iloc[5].to_dict()
    a = MockProvider().complete("Summarise.", task="summary", fields=row)
    b = MockProvider().complete("Summarise.", task="summary", fields=row)
    assert a.text == b.text


def test_response_carries_governance_metadata():
    response = MockProvider().complete("Summarise.", context="Call (555) 213-4455.")
    assert response.provider and response.model
    assert response.is_mock is True
    assert response.masked_kinds.get("PHONE") == 1


def test_base_provider_is_abstract():
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]
