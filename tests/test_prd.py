"""docs/PRD.md must carry the things the brief asked a PRD to carry.

Not a style check. Each assertion covers a commitment made elsewhere in the
codebase, so a PRD that quietly drops one fails the build.
"""

from __future__ import annotations

import re

from claimpilot import config

PRD = config.ROOT / "docs" / "PRD.md"


def _text() -> str:
    return PRD.read_text(encoding="utf-8")


def _normalised() -> str:
    return re.sub(r"\s+", " ", _text())


def test_prd_exists_and_is_short():
    """One to two pages. Long enough to be useful, short enough to be read."""
    assert PRD.exists()
    words = len(_text().split())
    assert 700 < words < 2600, f"PRD is {words} words"


def test_prd_covers_every_required_section():
    low = _text().lower()
    for heading in (
        "problem",
        "users",
        "goals",
        "non-goals",
        "mvp scope",
        "kpis",
        "risks",
        "phase 2 roadmap",
    ):
        assert heading in low, f"PRD is missing a {heading!r} section"


def test_prd_names_every_user_from_the_brief():
    low = _text().lower()
    for user in ("policyholder", "adjuster", "claims manager", "siu", "compliance"):
        assert user in low


def test_prd_states_the_no_auto_deny_rule():
    low = _normalised().lower()
    assert "never auto-deny" in low
    assert "auto-approve" in low


def test_prd_lists_the_roadmap_items_the_brief_asked_for():
    low = _text().lower()
    for item in (
        "property",
        "photo damage",
        "subrogation",
        "payments",
        "guidewire",
        "filings",
    ):
        assert item in low, f"roadmap is missing {item!r}"


def test_prd_declares_the_dataset_and_backtest_risks():
    low = _text().lower()
    assert "recalibration" in low or "recalibrat" in low
    assert "not evidence" in low or "assumption-driven" in low
    assert "reported" in low and "flag" in low


def test_prd_carries_the_disclaimer_and_claims_no_compliance():
    low = _normalised().lower()
    assert "not a licensed insurance product" in low
    assert "designed around" in low or "aligned with" in low
    assert "is compliant" not in low
    assert "certified" not in low or "not an attestation" in low


def test_prd_uses_only_the_fictional_carrier_name():
    text = _text()
    assert config.carrier_name() in text
    for real in ("Geico", "Progressive", "State Farm", "Allstate", "Nationwide"):
        assert real not in text


def test_prd_separates_guardrails_from_efficiency_kpis():
    low = _text().lower()
    assert "guardrail" in low
    assert "auto-denied" in low
