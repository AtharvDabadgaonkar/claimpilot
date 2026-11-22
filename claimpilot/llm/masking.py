"""PII masking — runs before any text leaves the process.

Data minimisation: an LLM only ever needs the *facts of loss*, never the
claimant's identity. Everything identifying is replaced with a stable
placeholder token before the prompt is built, and can be re-inserted
afterwards for display to the adjuster.

This is a demo-grade masker built on regular expressions. A real deployment
would use a validated PII detection service and would not rely on regex alone;
that gap is documented in docs/AI_GOVERNANCE.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- Patterns ---------------------------------------------------------------
# Order matters: more specific patterns run first so that, for example, an
# email is not partially consumed by the phone-number pattern.

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "POLICY",
        re.compile(
            r"\b(?:NWM[-\s]?\d[\d\-]{4,}|"
            r"(?:policy|pol\.?)\s*(?:number|no\.?|#)?\s*[:#]?\s*[A-Z]{0,4}\d{5,})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "VIN",
        re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"),
    ),
    (
        "PHONE",
        re.compile(
            r"(?<![\d\-])(?:\+?1[\s.\-]?)?"
            r"(?:\(\d{3}\)|\d{3})[\s.\-]\d{3}[\s.\-]\d{4}(?![\d\-])"
        ),
    ),
    (
        "SSN",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        "PLATE",
        re.compile(
            r"\b(?:license\s+plate|plate|tag)\s*(?:number|no\.?|#)?\s*[:#]?\s*"
            r"([A-Z0-9]{5,8})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ADDRESS",
        re.compile(
            r"\b\d{1,5}\s+(?:[A-Z0-9][A-Za-z0-9.]*\s+){0,3}"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|"
            r"Court|Ct|Way|Highway|Hwy|Place|Pl|Ridge|Trail)\b\.?",
            re.IGNORECASE,
        ),
    ),
]

# Honorific-led names ("Mr. John Smith") and explicit self-identification
# ("my name is Jane Doe"). Bare capitalised words are deliberately NOT masked:
# that would destroy place names and vehicle makes the triage needs.
_NAME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"
    ),
    # The trigger phrase matches case-insensitively via a scoped (?i:) group,
    # but the captured name must still be Capitalised. Applying IGNORECASE to
    # the whole pattern would match ordinary prose such as "I am not sure".
    re.compile(
        r"\b(?i:my name is|this is|contact|insured|claimant|driver"
        r"|policyholder)\s+(?:is\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+)\b"
    ),
]


@dataclass
class MaskResult:
    """Masked text plus the mapping needed to restore it for display."""

    text: str
    mapping: dict[str, str] = field(default_factory=dict)

    @property
    def masked_count(self) -> int:
        return len(self.mapping)

    @property
    def kinds(self) -> dict[str, int]:
        """How many items of each kind were masked, e.g. {'EMAIL': 1}."""
        counts: dict[str, int] = {}
        for token in self.mapping:
            kind = token.strip("[]").rsplit("_", 1)[0]
            counts[kind] = counts.get(kind, 0) + 1
        return counts

    def unmask(self, text: str) -> str:
        """Put the original values back, for display to an authorised user."""
        for token, original in self.mapping.items():
            text = text.replace(token, original)
        return text


def mask_text(text: str) -> MaskResult:
    """Replace personal data with stable placeholder tokens.

    Returns the masked text and a token -> original mapping. The same original
    value always maps to the same token within one call, so an LLM can still
    reason about "the same person" without ever seeing who they are.
    """
    if not text:
        return MaskResult(text=text or "", mapping={})

    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}
    counters: dict[str, int] = {}

    def token_for(kind: str, original: str) -> str:
        if original in reverse:
            return reverse[original]
        counters[kind] = counters.get(kind, 0) + 1
        token = f"[{kind}_{counters[kind]}]"
        mapping[token] = original
        reverse[original] = token
        return token

    masked = text

    # Named-entity style patterns with a capture group use the group where
    # present, so the leading phrase ("my name is") survives.
    for pattern in _NAME_PATTERNS:

        def _name_sub(match: re.Match[str]) -> str:
            captured = match.group(1) if match.groups() else None
            if captured:
                return match.group(0).replace(captured, token_for("NAME", captured))
            return token_for("NAME", match.group(0))

        masked = pattern.sub(_name_sub, masked)

    for kind, pattern in _PATTERNS:

        def _sub(match: re.Match[str], _kind: str = kind) -> str:
            captured = match.group(1) if match.groups() else None
            if captured:
                return match.group(0).replace(captured, token_for(_kind, captured))
            return token_for(_kind, match.group(0))

        masked = pattern.sub(_sub, masked)

    return MaskResult(text=masked, mapping=mapping)


def assert_masked(text: str) -> None:
    """Raise if obvious PII survives. Called as a last gate before egress."""
    leaks = []
    for kind, pattern in _PATTERNS:
        if kind in {"PLATE", "ADDRESS"}:
            continue  # these are heuristic; do not hard-fail on them
        if pattern.search(text):
            leaks.append(kind)
    if leaks:
        raise ValueError(
            "Refusing to send text to an LLM: unmasked "
            f"{', '.join(sorted(set(leaks)))} detected."
        )
