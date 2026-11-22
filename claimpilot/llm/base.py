"""LLM provider interface.

The app must work with no API key: tests, CI and the deployed Streamlit demo
all run against :class:`MockProvider`. Every provider goes through
:meth:`LLMProvider.complete`, which masks PII before the prompt is built and
refuses to send anything that still looks identifying.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from claimpilot.llm.masking import MaskResult, assert_masked, mask_text


@dataclass
class LLMResponse:
    """A completion plus the governance metadata the audit log needs."""

    text: str
    provider: str
    model: str
    is_mock: bool
    masked_kinds: dict[str, int] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def pii_items_masked(self) -> int:
        return sum(self.masked_kinds.values())


class LLMProvider(abc.ABC):
    """Base class. Subclasses implement :meth:`_generate` only."""

    name: str = "base"
    model: str = "none"
    is_mock: bool = True

    def complete(self, prompt: str, *, context: str = "", **kwargs: Any) -> LLMResponse:
        """Mask, gate, then generate.

        ``context`` carries claim text that may contain personal data. It is
        masked and the egress gate runs before ``_generate`` is reached, so no
        subclass can accidentally bypass masking.
        """
        masked: MaskResult = mask_text(context) if context else MaskResult("", {})
        if masked.text:
            assert_masked(masked.text)

        full_prompt = f"{prompt}\n\n{masked.text}".strip() if masked.text else prompt
        text = self._generate(full_prompt, **kwargs)

        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            is_mock=self.is_mock,
            masked_kinds=masked.kinds,
        )

    @abc.abstractmethod
    def _generate(self, prompt: str, **kwargs: Any) -> str:
        """Produce completion text from an already-masked prompt."""
        raise NotImplementedError
