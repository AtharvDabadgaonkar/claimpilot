"""LLM access layer: provider interface, mock fallback, and PII masking."""

from __future__ import annotations

import os
from pathlib import Path

from claimpilot.llm.base import LLMProvider, LLMResponse
from claimpilot.llm.masking import MaskResult, assert_masked, mask_text
from claimpilot.llm.mock import MockProvider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "MockProvider",
    "MaskResult",
    "mask_text",
    "assert_masked",
    "get_provider",
    "provider_badge",
]


def _secrets_file_exists() -> bool:
    """True if a Streamlit secrets file is present.

    Checked first because touching ``st.secrets`` outside a Streamlit runtime
    emits warnings, which would pollute test and CLI output.
    """
    from claimpilot.config import ROOT

    return (ROOT / ".streamlit" / "secrets.toml").exists() or Path(
        "~/.streamlit/secrets.toml"
    ).expanduser().exists()


def _read_secret(key: str) -> str | None:
    """Look for a key in Streamlit secrets, then the environment."""
    if _secrets_file_exists():
        try:
            import streamlit as st

            value = st.secrets.get(key)  # type: ignore[union-attr]
            if value:
                return str(value)
        except Exception:
            pass
    return os.environ.get(key) or None


def get_provider(force_mock: bool = False) -> LLMProvider:
    """Return the configured provider, or the mock when no key is available.

    Phase 1 ships the mock only. A hosted provider can be registered here
    later without any caller changing: every caller goes through
    ``LLMProvider.complete``, which masks PII first.
    """
    if force_mock:
        return MockProvider()
    api_key = _read_secret("ANTHROPIC_API_KEY")
    if not api_key:
        return MockProvider()
    # A hosted provider is not wired up yet (Phase 1 scope). Until it is, fall
    # back to the mock rather than silently failing at runtime.
    return MockProvider()


def provider_badge(provider: LLMProvider | None = None) -> str:
    """Short label for the UI so the AI mode in use is always visible."""
    provider = provider or get_provider()
    if provider.is_mock:
        return "Demo mode: rule-based AI"
    return f"LLM: {provider.name} ({provider.model})"
