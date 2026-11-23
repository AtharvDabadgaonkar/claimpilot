"""FNOL intake: schema validation and rule-based extraction."""

from __future__ import annotations

from datetime import date

import pytest
import yaml
from pydantic import ValidationError

from claimpilot import config
from claimpilot.intake import (
    CollisionType,
    ExtractedFNOL,
    FNOLSubmission,
    LossType,
    extract_fnol,
)

HELD_OUT = config.ROOT / "tests" / "fixtures" / "held_out_narratives.yaml"


# --- Submission schema --------------------------------------------------------


def test_valid_submission_accepted():
    s = FNOLSubmission(
        policy_number="NWM-123456",
        reported_date=date(2015, 2, 3),
        loss_date=date(2015, 2, 1),
        loss_type=LossType.SINGLE_VEHICLE,
        loss_city="Columbus",
        loss_state="oh",
        description="I hit a guardrail on the way home and the front is damaged.",
    )
    assert s.loss_state == "OH"


def test_loss_date_cannot_be_after_report_date():
    with pytest.raises(ValidationError, match="loss_date cannot be after"):
        FNOLSubmission(
            policy_number="NWM-123456",
            reported_date=date(2015, 2, 1),
            loss_date=date(2015, 2, 3),
            loss_type=LossType.SINGLE_VEHICLE,
            loss_city="Columbus",
            loss_state="OH",
            description="A description long enough to pass validation.",
        )


def test_description_must_be_substantive():
    with pytest.raises(ValidationError):
        FNOLSubmission(
            policy_number="NWM-123456",
            reported_date=date(2015, 2, 3),
            loss_date=date(2015, 2, 1),
            loss_type=LossType.SINGLE_VEHICLE,
            loss_city="Columbus",
            loss_state="OH",
            description="crash",
        )


# --- Extraction on the generated corpus ---------------------------------------


def test_extraction_returns_valid_schema_for_every_sample(narratives):
    """Required: extraction produces valid structured fields on the samples."""
    for text in narratives["narrative"].head(200):
        extracted = extract_fnol(text)
        assert isinstance(extracted, ExtractedFNOL)
        ExtractedFNOL.model_validate(extracted.model_dump())


def test_extraction_matches_ground_truth_on_every_narrative(narratives):
    """Scored against what the text states, not against the underlying row."""
    import pandas as pd

    checks = {
        "truth_loss_type": lambda e: e.loss_type.value,
        "truth_loss_date": lambda e: e.loss_date.isoformat() if e.loss_date else None,
        "truth_num_injured": lambda e: e.num_injured,
        "truth_police_report_available": lambda e: e.police_report_available,
        "truth_num_vehicles": lambda e: e.num_vehicles,
        "truth_witnesses": lambda e: e.witnesses,
        "truth_city": lambda e: e.city,
        "truth_state": lambda e: e.state,
        "truth_auto_make": lambda e: e.auto_make,
        "truth_auto_year": lambda e: e.auto_year,
    }
    mismatches = []
    for _, row in narratives.iterrows():
        extracted = extract_fnol(row["narrative"])
        for column, getter in checks.items():
            truth = row[column]
            if truth is not None and not isinstance(truth, bool | str) and pd.isna(truth):
                truth = None
            got = getter(extracted)
            if truth is None and got is None:
                continue
            if truth is None or got is None:
                mismatches.append((row["claim_id"], column, truth, got))
            elif isinstance(truth, int | float) and not isinstance(truth, bool):
                if int(truth) != int(got):
                    mismatches.append((row["claim_id"], column, truth, got))
            elif str(truth) != str(got):
                mismatches.append((row["claim_id"], column, truth, got))
    assert not mismatches, f"{len(mismatches)} mismatches, e.g. {mismatches[:3]}"


def test_extraction_is_deterministic(narratives):
    text = narratives.iloc[0]["narrative"]
    assert extract_fnol(text).model_dump() == extract_fnol(text).model_dump()


# --- Refusing to guess --------------------------------------------------------


def test_unstated_fields_stay_none():
    """Silence is a documentation gap, not a 'no'."""
    extracted = extract_fnol(
        "I skidded into a guardrail on January 5, 2015 in Hillsdale, PA. "
        "The front of my car is damaged."
    )
    assert extracted.police_report_available is None
    assert extracted.witnesses is None
    assert extracted.injury_reported is None


def test_vague_amount_is_not_turned_into_a_number():
    """The copilot must never invent a figure."""
    for phrase in (
        "It will cost a few thousand dollars.",
        "Maybe a couple thousand dollars.",
        "I have no idea what it will cost.",
        "Quite a lot, I am not sure how much.",
    ):
        extracted = extract_fnol(
            f"I was rear ended on February 2, 2015 in Columbus, OH. {phrase}"
        )
        assert extracted.estimated_amount_usd is None, phrase


def test_stated_amount_is_extracted_exactly():
    extracted = extract_fnol(
        "Rear collision on February 2, 2015 in Columbus, OH. "
        "The estimate is $18,750.00."
    )
    assert extracted.estimated_amount_usd == 18750.0


def test_missing_fields_are_reported_for_the_adjuster():
    extracted = extract_fnol(
        "Something happened to my car and it is damaged somehow, I think."
    )
    assert "police report status" in extracted.missing_fields()
    assert extracted.completeness < 0.5


# --- Robustness set -----------------------------------------------------------


def _held_out_cases():
    return yaml.safe_load(HELD_OUT.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _held_out_cases(), ids=lambda c: c["id"])
def test_robustness_cases(case):
    """Hand-written paraphrases that do not follow the generator's templates.

    These were used while tuning the extractor, so they are a regression suite
    rather than an unbiased held-out evaluation. See the caveat in the README.
    """
    extracted = extract_fnol(case["text"])
    for field, want in case["expect"].items():
        got = getattr(extracted, field)
        if hasattr(got, "value"):
            got = got.value
        if want is None:
            assert got is None, f"{case['id']}.{field}: expected None, got {got!r}"
        elif isinstance(want, float):
            assert got is not None and abs(got - want) < 1, f"{case['id']}.{field}"
        else:
            assert got == want, f"{case['id']}.{field}: expected {want!r}, got {got!r}"


def test_theft_and_parked_car_have_no_collision_direction():
    for text, expected in [
        ("My car was stolen overnight on March 1, 2015 in Arlington, NY.", LossType.THEFT),
        ("Someone hit my parked car on March 1, 2015 in Arlington, NY.", LossType.PARKED_CAR),
    ]:
        extracted = extract_fnol(text)
        assert extracted.loss_type is expected
        assert extracted.collision_type is CollisionType.NOT_APPLICABLE


def test_injury_count_drives_the_injury_flag():
    assert extract_fnol("Two people were injured.").injury_reported is True
    assert extract_fnol("Nobody was injured.").injury_reported is False
    assert extract_fnol("Nobody was hurt.").num_injured == 0
