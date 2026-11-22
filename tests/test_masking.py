"""PII masking is the gate before any text can reach an LLM."""

from __future__ import annotations

import pytest

from claimpilot.llm.masking import assert_masked, mask_text

STRICT_KINDS = ("EMAIL", "PHONE", "POLICY", "NAME", "SSN", "VIN")


@pytest.mark.parametrize(
    "text,kind",
    [
        ("Reach me at jordan.tran@example.com please.", "EMAIL"),
        ("Call (555) 213-4455 after six.", "PHONE"),
        ("Call 555-213-4455 after six.", "PHONE"),
        ("My policy number is NWM-342868.", "POLICY"),
        ("policy no: 521585", "POLICY"),
        ("My name is Jane Doe and I was driving.", "NAME"),
        ("The other driver was Mr. Robert Hale.", "NAME"),
        ("SSN 123-45-6789", "SSN"),
        ("VIN 1HGBH41JXMN109186", "VIN"),
        ("I live at 9935 4th Drive.", "ADDRESS"),
        ("License plate ABC1234", "PLATE"),
    ],
)
def test_each_pii_kind_is_masked(text, kind):
    result = mask_text(text)
    assert kind in result.kinds, f"{kind} not masked in: {text!r}"


def test_original_value_is_absent_from_masked_text():
    result = mask_text("Email me at jordan.tran@example.com.")
    assert "jordan.tran@example.com" not in result.text
    assert "[EMAIL_1]" in result.text


def test_repeated_value_gets_one_stable_token():
    """An LLM should still be able to reason about 'the same person'."""
    result = mask_text(
        "Contact Jane Doe. Jane Doe was driving. Mr. Robert Hale was the other party."
    )
    tokens = [t for t in result.mapping if t.startswith("[NAME")]
    assert len(set(result.mapping.values())) == len(tokens)
    assert result.text.count("[NAME_1]") >= 1


def test_unmask_round_trips():
    original = (
        "My name is Jane Doe, policy NWM-4821993, jane.doe@example.com, "
        "(555) 213-4455."
    )
    result = mask_text(original)
    assert result.unmask(result.text) == original


def test_claim_facts_survive_masking():
    """Masking must not destroy what triage needs to read."""
    text = (
        "Rear collision in Columbus, Ohio. A Toyota Camry hit my Honda Civic. "
        "Two people were injured and the police filed a report."
    )
    result = mask_text(text)
    assert result.masked_count == 0
    assert result.text == text


def test_ordinary_prose_is_not_mistaken_for_a_name():
    """Regression: 'I am not sure how much' once matched the name pattern."""
    result = mask_text("It will cost quite a lot, I am not sure how much.")
    assert "NAME" not in result.kinds


def test_no_false_positives_across_clean_narratives(narratives):
    """Measured over every generated narrative known to carry no PII."""
    clean = narratives[~narratives["truth_contains_pii"]]
    assert len(clean) > 100
    offenders = [
        t
        for t in clean["narrative"]
        if any(k in mask_text(t).kinds for k in STRICT_KINDS)
    ]
    assert not offenders, f"{len(offenders)} false positives, e.g. {offenders[:1]}"


def test_all_pii_narratives_are_detected(narratives):
    dirty = narratives[narratives["truth_contains_pii"]]
    assert len(dirty) > 100
    undetected = [t for t in dirty["narrative"] if mask_text(t).masked_count == 0]
    assert not undetected, f"{len(undetected)} narratives with undetected PII"


def test_egress_gate_passes_every_masked_narrative(narratives):
    for text in narratives["narrative"]:
        assert_masked(mask_text(text).text)


def test_egress_gate_rejects_unmasked_text():
    with pytest.raises(ValueError, match="unmasked"):
        assert_masked("Contact jane.doe@example.com for details.")


def test_empty_input_is_safe():
    assert mask_text("").text == ""
    assert mask_text("").masked_count == 0
