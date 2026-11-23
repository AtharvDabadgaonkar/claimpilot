"""Audit trail: justification requirements and override accounting."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from claimpilot import config
from claimpilot.audit import (
    AuditLog,
    AuditRecord,
    DecidedBy,
    DecisionType,
    reason_codes_for,
    seed_demo_log,
)


def base(**overrides):
    record = {
        "claim_id": "CLM-2015-00001",
        "ai_recommendation": "senior_adjuster",
        "decision": DecisionType.APPROVE,
    }
    record.update(overrides)
    return record


VALID_DENIAL_NOTE = "Policy lapsed 111 days before the date of loss; no cover in force."


# =============================================================================
# A denial cannot be submitted without justification.
# =============================================================================


def test_denial_without_a_reason_code_is_rejected():
    """Required: a denial can't be submitted without a reason code."""
    with pytest.raises(ValidationError):
        AuditRecord(**base(decision=DecisionType.DENY, note=VALID_DENIAL_NOTE))


def test_denial_without_a_note_is_rejected():
    with pytest.raises(ValidationError):
        AuditRecord(
            **base(decision=DecisionType.DENY, reason_code="DENY_NO_COVERAGE")
        )


def test_denial_with_a_trivial_note_is_rejected():
    with pytest.raises(ValidationError):
        AuditRecord(
            **base(
                decision=DecisionType.DENY,
                reason_code="DENY_NO_COVERAGE",
                note="no",
            )
        )


def test_denial_note_length_comes_from_config():
    minimum = int(config.assumption("workbench", "min_denial_note_chars"))
    short = "x" * (minimum - 1)
    with pytest.raises(ValidationError):
        AuditRecord(
            **base(decision=DecisionType.DENY, reason_code="DENY_OTHER", note=short)
        )
    record = AuditRecord(
        **base(decision=DecisionType.DENY, reason_code="DENY_OTHER", note="x" * minimum)
    )
    assert record.decision is DecisionType.DENY


def test_fully_justified_denial_is_accepted():
    record = AuditRecord(
        **base(
            decision=DecisionType.DENY,
            reason_code="DENY_NO_COVERAGE",
            note=VALID_DENIAL_NOTE,
        )
    )
    assert record.reason_code == "DENY_NO_COVERAGE"
    assert record.record_id


def test_an_invalid_reason_code_is_rejected():
    with pytest.raises(ValidationError):
        AuditRecord(
            **base(
                decision=DecisionType.DENY,
                reason_code="DENY_BECAUSE_I_SAID_SO",
                note=VALID_DENIAL_NOTE,
            )
        )


def test_a_reason_code_from_another_decision_type_is_rejected():
    """An escalation code cannot be used to justify a denial."""
    with pytest.raises(ValidationError):
        AuditRecord(
            **base(
                decision=DecisionType.DENY,
                reason_code="ESC_SUSPECTED_FRAUD",
                note=VALID_DENIAL_NOTE,
            )
        )


@pytest.mark.parametrize(
    "decision", [DecisionType.DENY, DecisionType.ESCALATE_SIU, DecisionType.REQUEST_INFO]
)
def test_adverse_and_deferring_decisions_require_a_reason_code(decision):
    with pytest.raises(ValidationError):
        AuditRecord(**base(decision=decision, note=VALID_DENIAL_NOTE))


def test_approval_needs_no_reason_code():
    record = AuditRecord(**base(decision=DecisionType.APPROVE))
    assert record.reason_code is None


def test_the_log_cannot_hold_an_unjustified_denial():
    """The guard is at schema level, so AuditLog cannot bypass it."""
    log = AuditLog()
    with pytest.raises(ValidationError):
        log.record_decision(**base(decision=DecisionType.DENY))
    assert len(log) == 0


def test_reason_codes_are_defined_for_every_decision_needing_one():
    for decision in DecisionType:
        if decision.requires_reason_code:
            assert reason_codes_for(decision), f"{decision} has no reason codes"


# =============================================================================
# Override accounting
# =============================================================================


@pytest.mark.parametrize(
    "recommendation,decision,expected",
    [
        ("stp_auto_approve", DecisionType.APPROVE, False),
        ("stp_auto_approve", DecisionType.DENY, True),
        ("siu_queue", DecisionType.ESCALATE_SIU, False),
        ("siu_queue", DecisionType.APPROVE, True),
        # Routing to a queue is not a substantive recommendation, so nothing
        # the adjuster then decides is an override.
        ("adjuster_queue", DecisionType.APPROVE, False),
        ("adjuster_queue", DecisionType.REQUEST_INFO, False),
        ("senior_adjuster", DecisionType.APPROVE, False),
        ("senior_adjuster", DecisionType.DENY, False),
    ],
)
def test_override_detection(recommendation, decision, expected):
    kwargs = base(ai_recommendation=recommendation, decision=decision)
    if decision.requires_reason_code:
        kwargs["reason_code"] = {
            DecisionType.DENY: "DENY_OTHER",
            DecisionType.ESCALATE_SIU: "ESC_OTHER",
            DecisionType.REQUEST_INFO: "INFO_OTHER",
        }[decision]
    if decision is DecisionType.DENY:
        kwargs["note"] = VALID_DENIAL_NOTE
    assert AuditRecord(**kwargs).is_override is expected


def test_system_decisions_are_never_overrides():
    record = AuditRecord(
        **base(
            ai_recommendation="stp_auto_approve",
            decision=DecisionType.APPROVE,
            decided_by=DecidedBy.SYSTEM,
        )
    )
    assert record.is_override is False


def test_override_rate_excludes_non_substantive_recommendations():
    """Regression: counting queue routings inflated the rate to 85%."""
    log = AuditLog()
    for _ in range(10):
        log.record_decision(
            **base(ai_recommendation="senior_adjuster", decision=DecisionType.APPROVE)
        )
    assert log.override_eligible_count == 0
    assert log.override_rate == 0.0

    log.record_decision(
        **base(ai_recommendation="siu_queue", decision=DecisionType.APPROVE)
    )
    assert log.override_eligible_count == 1
    assert log.override_rate == 1.0


# =============================================================================
# The record itself
# =============================================================================


def test_record_captures_the_ai_inputs_and_versions():
    record = AuditRecord(
        **base(
            ai_severity_band="High",
            ai_complexity_band="Medium",
            ai_fraud_band="Low",
            ai_fraud_probability=0.12,
            ai_reserve_point_usd=42_000.0,
            triage_version="triage-rules-v0.1",
            rules_version="routing-v0.1",
            fraud_model_version="fraud-hgb-v0.1",
            reserve_model_version="reserve-gbq-v0.1",
            inputs_used=["triage scores", "coverage check"],
        )
    )
    row = record.to_row()
    for key in (
        "timestamp_utc", "claim_id", "ai_recommendation", "ai_fraud_probability",
        "decision", "decided_by", "decider_id", "human_overrode_ai",
        "triage_version", "rules_version", "fraud_model_version", "inputs_used",
    ):
        assert key in row


def test_record_ids_are_unique_per_decision():
    log = AuditLog()
    for i in range(5):
        log.record_decision(**base(claim_id=f"CLM-{i}"))
    ids = [r.record_id for r in log]
    assert len(set(ids)) == len(ids)


def test_log_exports_to_csv_and_json():
    log = AuditLog()
    log.record_decision(**base())
    csv = log.to_csv()
    assert "claim_id" in csv and "human_overrode_ai" in csv
    assert log.to_json().startswith("[")


def test_empty_log_still_exports_a_header():
    assert "claim_id" in AuditLog().to_csv()


def test_filtering(scored_fixture=None):
    log = AuditLog()
    log.record_decision(**base(claim_id="A", decision=DecisionType.APPROVE))
    log.record_decision(
        **base(
            claim_id="B",
            decision=DecisionType.ESCALATE_SIU,
            reason_code="ESC_SUSPECTED_FRAUD",
        )
    )
    assert len(log.filter(decision=DecisionType.APPROVE)) == 1
    assert len(log.filter(claim_id="B")) == 1
    assert len(log.for_claim("A")) == 1


# =============================================================================
# Seeded demo history
# =============================================================================


def test_seeded_log_contains_system_and_human_decisions():
    from claimpilot.pipeline import load_scored

    log = seed_demo_log(load_scored())
    frame = log.to_frame()
    assert len(log) > 0
    assert set(frame["decided_by"]) == {"system", "human"}


def test_seeded_log_has_a_plausible_override_rate():
    from claimpilot.pipeline import load_scored

    log = seed_demo_log(load_scored())
    assert log.override_eligible_count > 0
    assert 0.0 < log.override_rate < 0.5, (
        "a demo override rate of 0% or above 50% would misrepresent the metric"
    )


def test_seeded_log_contains_no_denials():
    """Nothing in the seeded history denies a claim without a human doing it."""
    from claimpilot.pipeline import load_scored

    log = seed_demo_log(load_scored())
    system = [r for r in log if r.decided_by is DecidedBy.SYSTEM]
    assert all(r.decision is DecisionType.APPROVE for r in system)


# =============================================================================
# Friendly error messages (no tracebacks in the UI)
# =============================================================================


def test_validation_errors_are_explained_readably():
    """'1 validation error for AuditRecord' tells an adjuster nothing."""
    from pydantic import ValidationError

    from claimpilot.audit import explain_validation_error

    cases = [
        (base(decision=DecisionType.DENY), "requires a reason code"),
        (
            base(decision=DecisionType.DENY, reason_code="DENY_NO_COVERAGE"),
            "explanatory note",
        ),
        (
            base(
                decision=DecisionType.DENY,
                reason_code="ESC_OTHER",
                note=VALID_DENIAL_NOTE,
            ),
            "not a valid reason code",
        ),
    ]
    for kwargs, expected in cases:
        try:
            AuditRecord(**kwargs)
        except ValidationError as exc:
            message = explain_validation_error(exc)
            assert expected in message, f"got: {message}"
            assert "validation error for" not in message
            assert "Traceback" not in message
        else:
            raise AssertionError(f"expected a validation error for {kwargs}")
