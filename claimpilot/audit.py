"""Audit trail: every AI recommendation and every human decision.

Each record captures what the AI said, what the human did, and whether the
human overrode the AI — which is the single most useful governance signal a
claims AI programme produces.

Storage is in ``st.session_state`` because Streamlit Community Cloud resets
its file system on restart. That is a demo constraint, stated in the README
and in the UI: the log exports to CSV but does not persist across restarts.
A real deployment would write to an append-only store.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from claimpilot import config


class DecisionType(str, Enum):
    """What a human can do with a claim.

    ``APPROVE`` is also what the STP path records when it auto-approves.
    There is no automated denial: ``DENY`` is only ever written by a human,
    and the schema enforces that it carries a reason code and a note.
    """

    APPROVE = "approve"
    DENY = "deny"
    ESCALATE_SIU = "escalate"
    REQUEST_INFO = "request_info"

    @property
    def label(self) -> str:
        return {
            "approve": "Approved",
            "deny": "Denied",
            "escalate": "Escalated to SIU",
            "request_info": "Information requested",
        }[self.value]

    @property
    def requires_reason_code(self) -> bool:
        """Adverse and deferring decisions must be justified."""
        return self in (
            DecisionType.DENY,
            DecisionType.ESCALATE_SIU,
            DecisionType.REQUEST_INFO,
        )

    @property
    def is_adverse(self) -> bool:
        return self is DecisionType.DENY


class DecidedBy(str, Enum):
    HUMAN = "human"
    SYSTEM = "system"


def reason_codes_for(decision: DecisionType) -> dict[str, str]:
    """Valid reason codes for a decision type, from config."""
    mapping = config.assumption("workbench", "decision_reason_codes")
    key = {
        DecisionType.DENY: "deny",
        DecisionType.ESCALATE_SIU: "escalate",
        DecisionType.REQUEST_INFO: "request_info",
    }.get(decision)
    if key is None:
        return {}
    codes: dict[str, str] = {}
    for entry in mapping[key]:
        codes.update(entry)
    return codes


class MissingReasonCodeError(ValueError):
    """Raised when an adverse decision is submitted without justification."""


def explain_validation_error(exc: Exception) -> str:
    """Turn a pydantic ValidationError into one readable sentence.

    The raw text starts with "1 validation error for AuditRecord", which tells
    an adjuster nothing. The useful message is the wrapped ValueError.
    """
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        messages = []
        for error in exc.errors():
            message = str(error.get("msg", "")).replace("Value error, ", "").strip()
            if message:
                messages.append(message)
        if messages:
            return " ".join(messages)
    return str(exc).splitlines()[0]


class AuditRecord(BaseModel):
    """One immutable line in the audit trail."""

    record_id: str = ""
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    claim_id: str = Field(min_length=1)

    # What the AI recommended.
    ai_recommendation: str
    ai_severity_band: str | None = None
    ai_complexity_band: str | None = None
    ai_fraud_band: str | None = None
    ai_fraud_probability: float | None = None
    ai_reserve_point_usd: float | None = None
    triage_version: str | None = None
    rules_version: str | None = None
    fraud_model_version: str | None = None
    reserve_model_version: str | None = None
    inputs_used: list[str] = Field(default_factory=list)

    # What actually happened.
    decision: DecisionType
    decided_by: DecidedBy = DecidedBy.HUMAN
    decider_id: str = "adjuster-demo"
    reason_code: str | None = None
    note: str | None = None
    approved_amount_usd: float | None = None

    @field_validator("note")
    @classmethod
    def _strip_note(cls, v: str | None) -> str | None:
        return v.strip() if v else None

    @model_validator(mode="after")
    def _enforce_justification(self) -> AuditRecord:
        """A denial without a reason code and a note is not submittable.

        This is schema-level, not UI-level: the record cannot be constructed
        at all, so no code path can write an unjustified denial to the log.
        """
        if self.decision.requires_reason_code and not self.reason_code:
            raise MissingReasonCodeError(
                f"A '{self.decision.value}' decision requires a reason code."
            )

        if self.reason_code:
            valid = reason_codes_for(self.decision)
            if valid and self.reason_code not in valid:
                raise ValueError(
                    f"'{self.reason_code}' is not a valid reason code for "
                    f"'{self.decision.value}'. Valid: {sorted(valid)}"
                )

        if self.decision.is_adverse:
            minimum = int(config.assumption("workbench", "min_denial_note_chars"))
            if not self.note or len(self.note) < minimum:
                raise MissingReasonCodeError(
                    f"A denial requires an explanatory note of at least "
                    f"{minimum} characters."
                )

        if not self.record_id:
            seed = (
                f"{self.claim_id}|{self.timestamp_utc.isoformat()}|"
                f"{self.decision.value}|{self.decider_id}"
            )
            self.record_id = hashlib.sha256(seed.encode()).hexdigest()[:16]
        return self

    @property
    def is_override(self) -> bool:
        """Did the human contradict a substantive AI recommendation?

        Routing a claim to the adjuster or senior-adjuster queue is not a
        recommendation to approve or deny — it is the AI saying "a human
        should look at this". Whatever the adjuster then decides, nothing has
        been overridden. Counting those as overrides inflated the rate to 85%
        and would have made the governance metric meaningless.

        Only two routings carry a substantive call:
          * STP auto-approve  -> anything other than approval is an override
          * SIU referral      -> not escalating is an override
        """
        if self.decided_by is DecidedBy.SYSTEM:
            return False

        recommended = _normalise(self.ai_recommendation)
        if recommended not in _SUBSTANTIVE_RECOMMENDATIONS:
            return False
        return recommended != self.decision.value

    def to_row(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "timestamp_utc": self.timestamp_utc.isoformat(timespec="seconds"),
            "claim_id": self.claim_id,
            "ai_recommendation": self.ai_recommendation,
            "ai_severity_band": self.ai_severity_band,
            "ai_complexity_band": self.ai_complexity_band,
            "ai_fraud_band": self.ai_fraud_band,
            "ai_fraud_probability": self.ai_fraud_probability,
            "ai_reserve_point_usd": self.ai_reserve_point_usd,
            "decision": self.decision.value,
            "decision_label": self.decision.label,
            "decided_by": self.decided_by.value,
            "decider_id": self.decider_id,
            "reason_code": self.reason_code,
            "note": self.note,
            "approved_amount_usd": self.approved_amount_usd,
            "human_overrode_ai": self.is_override,
            "triage_version": self.triage_version,
            "rules_version": self.rules_version,
            "fraud_model_version": self.fraud_model_version,
            "reserve_model_version": self.reserve_model_version,
            "inputs_used": ", ".join(self.inputs_used),
        }


# Map a routing destination to the decision it effectively recommends, so an
# override can be detected. "review" means the AI made no substantive call.
_RECOMMENDATION_ALIASES = {
    "stp_auto_approve": "approve",
    "siu_queue": "escalate",
    "adjuster_queue": "review",
    "senior_adjuster": "review",
}

# Only these recommendations are specific enough to be overridden.
_SUBSTANTIVE_RECOMMENDATIONS = {"approve", "escalate"}


def _normalise(value: str | None) -> str:
    if not value:
        return ""
    return _RECOMMENDATION_ALIASES.get(value.lower(), value.lower())


class AuditLog:
    """An append-only collection of audit records."""

    def __init__(self, records: list[AuditRecord] | None = None) -> None:
        self._records: list[AuditRecord] = list(records or [])

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    @property
    def records(self) -> list[AuditRecord]:
        return list(self._records)

    def append(self, record: AuditRecord) -> AuditRecord:
        self._records.append(record)
        return record

    def record_decision(self, **kwargs: Any) -> AuditRecord:
        """Build and append a record. Raises if the decision is unjustified."""
        return self.append(AuditRecord(**kwargs))

    def to_frame(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame(columns=list(AuditRecord.model_fields) + ["human_overrode_ai"])
        return pd.DataFrame([r.to_row() for r in self._records])

    def to_csv(self) -> str:
        buffer = io.StringIO()
        self.to_frame().to_csv(buffer, index=False)
        return buffer.getvalue()

    def to_json(self) -> str:
        return json.dumps([r.to_row() for r in self._records], indent=2, default=str)

    @property
    def override_rate(self) -> float:
        """How often humans contradicted a substantive AI recommendation.

        The denominator is only those human decisions where the AI actually
        made a call (fast-track or SIU referral). Claims merely routed to a
        queue for review are excluded, because there is nothing to override.
        """
        eligible = [
            r
            for r in self._records
            if r.decided_by is DecidedBy.HUMAN
            and _normalise(r.ai_recommendation) in _SUBSTANTIVE_RECOMMENDATIONS
        ]
        if not eligible:
            return 0.0
        return sum(r.is_override for r in eligible) / len(eligible)

    @property
    def override_eligible_count(self) -> int:
        """Human decisions where the AI made a substantive recommendation."""
        return len(
            [
                r
                for r in self._records
                if r.decided_by is DecidedBy.HUMAN
                and _normalise(r.ai_recommendation) in _SUBSTANTIVE_RECOMMENDATIONS
            ]
        )

    def for_claim(self, claim_id: str) -> list[AuditRecord]:
        return [r for r in self._records if r.claim_id == claim_id]

    def filter(
        self,
        decision: DecisionType | None = None,
        claim_id: str | None = None,
        overrides_only: bool = False,
    ) -> list[AuditRecord]:
        results = self._records
        if decision is not None:
            results = [r for r in results if r.decision is decision]
        if claim_id:
            results = [r for r in results if r.claim_id == claim_id]
        if overrides_only:
            results = [r for r in results if r.is_override]
        return results


def seed_demo_log(scored: pd.DataFrame, limit: int = 160) -> AuditLog:
    """Seed the log so the demo opens with history rather than an empty table.

    Records the system's own STP auto-approvals plus a deterministic sample of
    human decisions, including a few overrides so the override rate is not
    trivially zero.
    """
    log = AuditLog()
    auto = scored[scored["auto_approved"]].head(limit // 2)

    for _, row in auto.iterrows():
        log.record_decision(
            claim_id=row["claim_id"],
            ai_recommendation=row["destination"],
            ai_severity_band=row["severity_band"],
            ai_complexity_band=row["complexity_band"],
            ai_fraud_band=row["fraud_band"],
            ai_fraud_probability=float(row["fraud_probability"]),
            triage_version=row["triage_version"],
            rules_version=row["rules_version"],
            decision=DecisionType.APPROVE,
            decided_by=DecidedBy.SYSTEM,
            decider_id="claimpilot-stp",
            inputs_used=["triage scores", "coverage check"],
        )

    # A deterministic slice of human decisions, including deliberate overrides.
    others = scored[~scored["auto_approved"]].head(limit // 2)
    for i, (_, row) in enumerate(others.iterrows()):
        recommended = row["destination"]
        if i % 8 == 0 and recommended == "siu_queue":
            # Adjuster disagrees with the SIU referral and approves instead.
            # Roughly 1 in 8, which puts the demo override rate in a plausible
            # range rather than at an alarming or implausibly perfect value.
            decision, code, note = DecisionType.APPROVE, None, "Reviewed; indicators explained by the circumstances."
        elif recommended == "siu_queue":
            decision, code, note = (
                DecisionType.ESCALATE_SIU,
                "ESC_SUSPECTED_FRAUD",
                "Referred to SIU in line with the triage recommendation.",
            )
        elif i % 4 == 0:
            decision, code, note = (
                DecisionType.REQUEST_INFO,
                "INFO_POLICE_REPORT",
                "Police report outstanding; requested from the insured.",
            )
        else:
            decision, code, note = DecisionType.APPROVE, None, "Coverage confirmed; settled within authority."

        log.record_decision(
            claim_id=row["claim_id"],
            ai_recommendation=recommended,
            ai_severity_band=row["severity_band"],
            ai_complexity_band=row["complexity_band"],
            ai_fraud_band=row["fraud_band"],
            ai_fraud_probability=float(row["fraud_probability"]),
            triage_version=row["triage_version"],
            rules_version=row["rules_version"],
            decision=decision,
            decided_by=DecidedBy.HUMAN,
            decider_id=f"adjuster-{(i % 4) + 1:02d}",
            reason_code=code,
            note=note,
            inputs_used=["triage scores", "coverage check", "reserve suggestion"],
        )
    return log
