"""Digital FNOL intake: structured schemas and rule-based extraction.

A policyholder submits a short form plus a free-text account of what happened.
The extractor turns that free text into structured fields, which pydantic
validates before anything downstream reads them.

In demo mode the extractor is **rules and regular expressions**, not a language
model. That is a deliberate trade-off: it is deterministic, needs no API key,
and cannot invent a value. It is also tuned to the phrasing of the generated
narratives, so the accuracy reported in reports/metrics.json measures *this
extractor on this corpus* and would not transfer unchanged to real FNOL text.
That limitation is stated in the README and the model card.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------


class LossType(str, Enum):
    SINGLE_VEHICLE = "Single Vehicle Collision"
    MULTI_VEHICLE = "Multi-vehicle Collision"
    THEFT = "Vehicle Theft"
    PARKED_CAR = "Parked Car"
    UNKNOWN = "Unknown"


class CollisionType(str, Enum):
    REAR = "Rear Collision"
    SIDE = "Side Collision"
    FRONT = "Front Collision"
    NOT_APPLICABLE = "Not applicable"


class DamageArea(str, Enum):
    FRONT = "Front"
    REAR = "Rear"
    SIDE = "Side"
    UNDISCLOSED = "Undisclosed"


# ---------------------------------------------------------------------------
# What the policyholder submits
# ---------------------------------------------------------------------------


class FNOLSubmission(BaseModel):
    """The raw first notice of loss, as filed."""

    policy_number: str = Field(min_length=3, max_length=32)
    reported_date: date
    loss_date: date
    loss_type: LossType
    loss_city: str = Field(min_length=1, max_length=80)
    loss_state: str = Field(min_length=2, max_length=2)
    description: str = Field(min_length=20, max_length=5000)
    photo_filename: str | None = None

    @field_validator("loss_state")
    @classmethod
    def _upper_state(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _loss_not_in_future(self) -> FNOLSubmission:
        if self.loss_date > self.reported_date:
            raise ValueError("loss_date cannot be after reported_date")
        return self


# ---------------------------------------------------------------------------
# What the extractor returns
# ---------------------------------------------------------------------------


class ExtractedFNOL(BaseModel):
    """Structured fields pulled out of the free-text description.

    Every field is optional: a claimant who does not mention the police is not
    the same as a claimant who says no police came. ``None`` means "not stated",
    and downstream triage reads that as a documentation gap rather than a 'no'.
    """

    loss_date: date | None = None
    loss_type: LossType = LossType.UNKNOWN
    collision_type: CollisionType | None = None
    damage_areas: list[DamageArea] = Field(default_factory=list)
    injury_reported: bool | None = None
    num_injured: int | None = Field(default=None, ge=0, le=50)
    police_report_available: bool | None = None
    num_vehicles: int | None = Field(default=None, ge=1, le=50)
    witnesses: int | None = Field(default=None, ge=0, le=50)
    city: str | None = None
    state: str | None = None
    auto_make: str | None = None
    auto_year: int | None = Field(default=None, ge=1900, le=2100)
    estimated_amount_usd: float | None = Field(default=None, ge=0)
    parties_mentioned: int = 1

    # Provenance of the extraction itself, for the audit trail.
    extractor_version: str = "rules-v0.1"
    fields_found: int = 0
    fields_attempted: int = 15

    @model_validator(mode="after")
    def _injury_count_agrees_with_flag(self) -> ExtractedFNOL:
        if self.num_injured is not None:
            self.injury_reported = self.num_injured > 0
        return self

    @property
    def completeness(self) -> float:
        """Share of fields the extractor managed to populate."""
        return round(self.fields_found / self.fields_attempted, 3)

    def missing_fields(self) -> list[str]:
        """Fields a human will need to chase before the claim can be decided."""
        checks = {
            "loss date": self.loss_date,
            "loss type": None if self.loss_type is LossType.UNKNOWN else True,
            "police report status": self.police_report_available,
            "injury status": self.injury_reported,
            "estimated amount": self.estimated_amount_usd,
            "location": self.city,
        }
        return [name for name, value in checks.items() if value is None]


# ---------------------------------------------------------------------------
# Rule-based extraction
# ---------------------------------------------------------------------------

_NUMBER_WORDS = {
    "no": 0, "nobody": 0, "none": 0, "one": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "a": 1, "an": 1,
}

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December"
)

_DATE_PATTERNS = [
    re.compile(rf"\b({_MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.IGNORECASE),
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
    re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"),
]

_LOSS_TYPE_CUES: list[tuple[LossType, re.Pattern[str]]] = [
    (
        LossType.THEFT,
        re.compile(
            r"\b(stolen|theft|someone took my (vehicle|car)|(my|the) (car|vehicle) was gone)\b",
            re.IGNORECASE,
        ),
    ),
    (
        LossType.PARKED_CAR,
        re.compile(
            r"\b(parked|unattended|hit my parked|while parked)\b", re.IGNORECASE
        ),
    ),
    (
        LossType.MULTI_VEHICLE,
        re.compile(
            r"\b(multi[- ]vehicle|pile[- ]?up|chain[- ]reaction|rear[- ]ended|"
            r"(\d+|two|three|four|five)\s+(vehicles|cars)\s+(were|was)\s+involved|"
            r"another car|third vehicle|second vehicle)\b",
            re.IGNORECASE,
        ),
    ),
    (
        LossType.SINGLE_VEHICLE,
        re.compile(
            r"\b(lost control|went off the road|skidded|collided with|hit a)\b",
            re.IGNORECASE,
        ),
    ),
]

_COLLISION_CUES: list[tuple[CollisionType, DamageArea, re.Pattern[str]]] = [
    (
        CollisionType.REAR,
        DamageArea.REAR,
        re.compile(r"\b(rear[- ]end(?:ed)?|rear of my vehicle|hit from behind|rear collision)\b", re.IGNORECASE),
    ),
    (
        CollisionType.SIDE,
        DamageArea.SIDE,
        re.compile(r"\b(struck on the side|side collision|t-boned|sideswip(?:e|ed))\b", re.IGNORECASE),
    ),
    (
        CollisionType.FRONT,
        DamageArea.FRONT,
        re.compile(r"\b(front end took|head[- ]on|front collision|front of my (car|vehicle))\b", re.IGNORECASE),
    ),
]

_NO_INJURY = re.compile(
    r"\b(nobody|no one|no-one)\s+(?:was\s+)?(?:injured|hurt)\b|\bno injuries\b",
    re.IGNORECASE,
)
_INJURY_COUNT = re.compile(
    r"\b(\d+|one|two|three|four|five)\s+(?:person|people)\s+(?:was|were)\s+injured",
    re.IGNORECASE,
)
_INJURY_GENERIC = re.compile(
    r"\b(injured|injury|hurt|medical attention|ambulance|hospital)\b", re.IGNORECASE
)

_POLICE_YES = re.compile(
    r"\b(police (attended|came|arrived)|report was filed|filed a (police )?report|"
    r"police (took|made) a report|police report (is )?available)\b",
    re.IGNORECASE,
)
_POLICE_NO = re.compile(
    r"\b(no police report|police (were|was) not called|did not call the police|"
    r"no report was taken|(?:police|cops)\s+(?:did\s?n[o']?t|never)\s+(?:come|attend|show))\b",
    re.IGNORECASE,
)

_VEHICLE_COUNT = re.compile(
    r"\b(\d+|two|three|four|five)\s+(?:vehicles|cars)\b", re.IGNORECASE
)

# "pushed into a third vehicle" states the total implicitly: the insured plus
# the vehicles ahead of it. The ordinal gives the count directly.
_VEHICLE_ORDINAL = re.compile(
    r"\b(?:into|hit)\s+an?\s+(second|third|fourth|fifth)\s+(?:vehicle|car)\b",
    re.IGNORECASE,
)
_ORDINAL_TO_COUNT = {"second": 2, "third": 3, "fourth": 4, "fifth": 5}
_WITNESS_COUNT = re.compile(
    r"\b(\d+|one|two|three|four|five|no)\s+witness(?:es)?\b", re.IGNORECASE
)
_NO_WITNESS = re.compile(r"\b(no witnesses|nobody saw|there were no witnesses)\b", re.IGNORECASE)

_LOCATION_PATTERNS = [
    re.compile(r"\bin\s+([A-Z][A-Za-z\s.'-]{1,40}?),\s*([A-Z]{2})\b"),
    # "March 1, 2015, Arlington, NY." — a bare "City, ST" with no preposition.
    re.compile(r",\s*([A-Z][A-Za-z.'-]{1,30}(?:\s[A-Z][A-Za-z.'-]{1,30})?),\s*([A-Z]{2})\b"),
]

# Known makes in this book of business. Matching against a closed list
# (case-insensitively) is more robust than requiring claimant capitalisation.
_KNOWN_MAKES = (
    "Accura", "Audi", "BMW", "Chevrolet", "Dodge", "Ford", "Honda", "Jeep",
    "Mercedes", "Nissan", "Saab", "Suburu", "Toyota", "Volkswagen",
)
_MAKE_CANONICAL = {m.lower(): m for m in _KNOWN_MAKES}

_VEHICLE_YEAR_MAKE = re.compile(
    r"\b((?:19|20)\d{2})\s+(" + "|".join(_KNOWN_MAKES) + r")\b",
    re.IGNORECASE,
)
_VEHICLE_GENERIC = re.compile(
    r"\b((?:19|20)\d{2})\s+([A-Z][A-Za-z]+)(?:\s+[A-Za-z0-9]+)?\b"
)

# Dollar amounts. Deliberately refuses "a few thousand dollars" — a vague
# claimant must leave the field empty rather than have a number guessed for them.
_AMOUNT = re.compile(r"\$\s?([\d,]+(?:\.\d{2})?)\b")

_VAGUE_AMOUNT = re.compile(
    r"\b(a few|a couple|several|quite a lot|not sure|no idea)\b", re.IGNORECASE
)


def _word_to_int(token: str) -> int | None:
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


def _extract_date(text: str) -> date | None:
    for pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        try:
            if pattern is _DATE_PATTERNS[0]:
                return datetime.strptime(
                    f"{match.group(1)} {match.group(2)} {match.group(3)}", "%B %d %Y"
                ).date()
            if pattern is _DATE_PATTERNS[1]:
                return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            return date(int(match.group(3)), int(match.group(1)), int(match.group(2)))
        except ValueError:
            continue
    return None


def extract_fnol(text: str) -> ExtractedFNOL:
    """Pull structured fields out of a free-text loss description.

    Returns ``None`` for anything not clearly stated. Nothing is guessed.
    """
    data: dict[str, Any] = {}
    found = 0

    if (loss_date := _extract_date(text)) is not None:
        data["loss_date"] = loss_date
        found += 1

    # Loss type: first cue wins, most specific first.
    for loss_type, pattern in _LOSS_TYPE_CUES:
        if pattern.search(text):
            data["loss_type"] = loss_type
            found += 1
            break

    # Collision direction and damage area.
    for collision_type, area, pattern in _COLLISION_CUES:
        if pattern.search(text):
            data["collision_type"] = collision_type
            data["damage_areas"] = [area]
            found += 2
            break
    else:
        if data.get("loss_type") in (LossType.THEFT, LossType.PARKED_CAR):
            data["collision_type"] = CollisionType.NOT_APPLICABLE
            found += 1

    # Injuries.
    if match := _INJURY_COUNT.search(text):
        count = _word_to_int(match.group(1))
        if count is not None:
            data["num_injured"] = count
            data["injury_reported"] = count > 0
            found += 2
    elif _NO_INJURY.search(text):
        data["num_injured"] = 0
        data["injury_reported"] = False
        found += 2
    elif _INJURY_GENERIC.search(text):
        data["injury_reported"] = True
        found += 1

    # Police report. An explicit "no" is information; silence is not.
    if _POLICE_NO.search(text):
        data["police_report_available"] = False
        found += 1
    elif _POLICE_YES.search(text):
        data["police_report_available"] = True
        found += 1

    # Vehicles involved.
    if match := _VEHICLE_COUNT.search(text):
        count = _word_to_int(match.group(1))
        if count is not None:
            data["num_vehicles"] = count
            found += 1
    elif match := _VEHICLE_ORDINAL.search(text):
        data["num_vehicles"] = _ORDINAL_TO_COUNT[match.group(1).lower()]
        found += 1
    elif data.get("loss_type") in (
        LossType.SINGLE_VEHICLE,
        LossType.THEFT,
        LossType.PARKED_CAR,
    ):
        # Only the insured vehicle is on the claim in each of these cases, so
        # one vehicle is derivable from the loss type rather than guessed.
        data["num_vehicles"] = 1
        found += 1

    # Witnesses.
    if _NO_WITNESS.search(text):
        data["witnesses"] = 0
        found += 1
    elif match := _WITNESS_COUNT.search(text):
        count = _word_to_int(match.group(1))
        if count is not None:
            data["witnesses"] = count
            found += 1

    # Location.
    for pattern in _LOCATION_PATTERNS:
        if match := pattern.search(text):
            data["city"] = match.group(1).strip()
            data["state"] = match.group(2)
            found += 2
            break

    # Vehicle year and make.
    if match := _VEHICLE_YEAR_MAKE.search(text):
        data["auto_year"] = int(match.group(1))
        data["auto_make"] = _MAKE_CANONICAL[match.group(2).lower()]
        found += 2
    elif match := _VEHICLE_GENERIC.search(text):
        data["auto_year"] = int(match.group(1))
        data["auto_make"] = match.group(2)
        found += 2

    # Amount — only when a figure is actually given.
    if match := _AMOUNT.search(text):
        try:
            data["estimated_amount_usd"] = float(match.group(1).replace(",", ""))
            found += 1
        except ValueError:
            pass

    # Parties: how many people the narrative implicates.
    vehicles = data.get("num_vehicles") or 1
    data["parties_mentioned"] = max(1, int(vehicles))

    data["fields_found"] = found
    return ExtractedFNOL(**data)


def submission_from_narrative(
    narrative: str, policy_number: str, reported_date: date
) -> tuple[FNOLSubmission, ExtractedFNOL]:
    """Build a validated submission from a narrative, as the demo form does."""
    extracted = extract_fnol(narrative)
    submission = FNOLSubmission(
        policy_number=policy_number,
        reported_date=reported_date,
        loss_date=extracted.loss_date or reported_date,
        loss_type=extracted.loss_type,
        loss_city=extracted.city or "Unknown",
        loss_state=extracted.state or "XX",
        description=narrative,
    )
    return submission, extracted
