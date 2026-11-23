"""Adjuster workbench: prioritised queue, copilot briefing, decisions."""

from __future__ import annotations

import streamlit as st

from claimpilot.audit import (
    DecidedBy,
    DecisionType,
    MissingReasonCodeError,
    explain_validation_error,
    reason_codes_for,
)
from claimpilot.copilot import build_briefing
from claimpilot.fraud_model import MODEL_VERSION as FRAUD_VERSION
from claimpilot.pipeline import load_scored
from claimpilot.ui import (
    footer,
    friendly_error,
    get_audit_log,
    how_calculated,
    llm_provider,
    load_tables,
    page_setup,
    reserve_bundle,
    simulated_badge,
    storage_warning,
)
from claimpilot.workbench import build_queue, demo_today

ADJUSTER_DESTINATIONS = ["adjuster_queue", "senior_adjuster"]

page_setup(
    "Adjuster Workbench",
    "A prioritised queue, an AI briefing on each claim, and decisions that "
    "all land in the audit trail.",
)

try:
    tables = load_tables()
    scored = load_scored()
    queue = build_queue(
        scored, tables["claims"], tables["deadlines"], ADJUSTER_DESTINATIONS
    )
    log = get_audit_log()
except Exception as exc:  # pragma: no cover - UI guard
    friendly_error(exc, "build the adjuster queue")
    st.stop()

st.caption(f"🗓️ Demo date: **{demo_today()}** — the dataset's losses fall in Jan–Mar 2015.")

# --- Queue --------------------------------------------------------------------
st.subheader("Your queue")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Claims in queue", f"{len(queue):,}")
k2.metric("Deadline breached", f"{(queue['deadline_status'] == 'Breached').sum():,}")
k3.metric("At risk", f"{(queue['deadline_status'] == 'At risk').sum():,}")
k4.metric("Decisions logged", f"{len(log):,}")
how_calculated(
    "priority = weighted sum of severity, deadline status, fraud and "
    "complexity bands, open coverage issues and claim age, using the weights "
    "in config/assumptions.yaml. Higher is more urgent."
)
simulated_badge("Priority weights and deadlines are simulated and illustrative")

status_filter = st.multiselect(
    "Filter by deadline status",
    ["Breached", "At risk", "On track"],
    default=["Breached", "At risk", "On track"],
)
view = queue[queue["deadline_status"].isin(status_filter)]

st.dataframe(
    view[
        [
            "claim_id",
            "priority_score",
            "deadline_status",
            "days_to_decide_due",
            "severity_band",
            "complexity_band",
            "fraud_band",
            "coverage_status",
            "total_claim_amount",
            "destination_label",
        ]
    ].head(50),
    hide_index=True,
    use_container_width=True,
    column_config={
        "claim_id": "Claim",
        "priority_score": st.column_config.NumberColumn("Priority", format="%.1f"),
        "deadline_status": "Deadline",
        "days_to_decide_due": st.column_config.NumberColumn("Days to decide"),
        "severity_band": "Severity",
        "complexity_band": "Complexity",
        "fraud_band": "Fraud",
        "coverage_status": "Coverage",
        "total_claim_amount": st.column_config.NumberColumn("Claimed", format="$%d"),
        "destination_label": "Routed to",
    },
)
st.caption("Showing the top 50 by priority.")

if view.empty:
    st.info("No claims match that filter.", icon="ℹ️")
    footer()
    st.stop()

# --- Selected claim -----------------------------------------------------------
st.divider()
st.subheader("Work a claim")

claim_id = st.selectbox("Claim", view["claim_id"].head(50).tolist())
row = view.loc[view["claim_id"] == claim_id].iloc[0]
claim_full = tables["claims"].loc[tables["claims"]["claim_id"] == claim_id].iloc[0]
narrative_rows = tables["narratives"].loc[
    tables["narratives"]["claim_id"] == claim_id, "narrative"
]
narrative = narrative_rows.iloc[0] if len(narrative_rows) else ""

briefing = build_briefing(
    {**claim_full.to_dict(), **row.to_dict()},
    provider=llm_provider(),
    reserve_bundle=reserve_bundle(),
    narrative=narrative,
    triage_reasons={
        "Severity": list(row["severity_reasons"]),
        "Complexity": list(row["complexity_reasons"]),
        "Fraud signal": list(row["fraud_reasons"]),
    },
)

left, right = st.columns([3, 2])

with left:
    st.markdown("#### AI briefing")
    st.info(briefing.summary, icon="📝")
    st.markdown("**Next best action**")
    st.write(briefing.next_best_action)

    st.markdown("**Missing documents**")
    if briefing.missing_documents:
        for item in briefing.missing_documents:
            st.markdown(f"- **{item['document']}** — {item['why']}")
    else:
        st.write("Nothing outstanding.")

    st.markdown("**Why it scored this way**")
    for label, reasons in briefing.triage_reasons.items():
        with st.expander(f"{label}: {row[label.split()[0].lower() + '_band']}"):
            for reason in reasons:
                st.markdown(f"- {reason}")

with right:
    st.markdown("#### Reserve suggestion")
    st.metric("Suggested reserve", f"${briefing.reserve.point_usd:,.0f}")
    st.caption(
        f"Range ${briefing.reserve.low_usd:,.0f} – ${briefing.reserve.high_usd:,.0f} "
        "(10th–90th percentile)"
    )
    how_calculated(
        "GradientBoosting quantile regression on incident and policy "
        "characteristics. The claim-amount components are excluded as target "
        "leakage. **This figure comes from the model, never from the language "
        "model.**"
    )
    st.divider()
    st.markdown("#### Priority")
    st.metric("Score", f"{row['priority_score']:.1f}")
    for reason in row["priority_reasons"]:
        st.caption(f"• {reason}")
    st.divider()
    st.markdown("#### Claim")
    st.write(f"**Claimed:** ${row['total_claim_amount']:,.0f}")
    st.write(f"**Incident:** {row['incident_type']} — {row['incident_severity']}")
    st.write(f"**Coverage:** {row['coverage_status']}")
    st.write(f"**Deadline:** {row['deadline_status']} ({row['days_to_decide_due']} days to decide)")

# --- Decision -----------------------------------------------------------------
st.divider()
st.subheader("Decide")

st.warning(
    "**The system can auto-approve but never auto-deny.** A denial is always a "
    "human act, and this form will not accept one without a reason code and a "
    "substantive note.",
    icon="🔒",
)

decision_choice = st.radio(
    "Decision",
    list(DecisionType),
    format_func=lambda d: d.label,
    horizontal=True,
)

codes = reason_codes_for(decision_choice)
reason_code = None
if codes:
    reason_code = st.selectbox(
        "Reason code (required)",
        list(codes),
        format_func=lambda c: f"{c} — {codes[c]}",
    )
elif decision_choice is DecisionType.APPROVE:
    st.caption("No reason code required to approve.")

note = st.text_area(
    "Note"
    + (" (required, min 20 characters)" if decision_choice is DecisionType.DENY else ""),
    placeholder="Explain the decision for the file…",
)

approved_amount = None
if decision_choice is DecisionType.APPROVE:
    approved_amount = st.number_input(
        "Approved amount (USD)",
        min_value=0.0,
        value=float(briefing.reserve.point_usd),
        step=500.0,
        help="Defaults to the model's reserve suggestion; the adjuster sets the figure.",
    )

if st.button("Submit decision", type="primary"):
    try:
        record = log.record_decision(
            claim_id=claim_id,
            ai_recommendation=row["destination"],
            ai_severity_band=row["severity_band"],
            ai_complexity_band=row["complexity_band"],
            ai_fraud_band=row["fraud_band"],
            ai_fraud_probability=float(row["fraud_probability"]),
            ai_reserve_point_usd=float(briefing.reserve.point_usd),
            triage_version=row["triage_version"],
            rules_version=row["rules_version"],
            fraud_model_version=FRAUD_VERSION,
            reserve_model_version=briefing.reserve.model_version,
            inputs_used=["triage scores", "coverage check", "reserve model", "claim record"],
            decision=decision_choice,
            decided_by=DecidedBy.HUMAN,
            reason_code=reason_code,
            note=note or None,
            approved_amount_usd=approved_amount,
        )
    except (MissingReasonCodeError, ValueError) as exc:
        st.error(
            f"That decision can't be submitted. {explain_validation_error(exc)}",
            icon="🚫",
        )
    else:
        st.success(
            f"{record.decision.label} — logged as `{record.record_id}`"
            + (" (recorded as an override of the AI recommendation)" if record.is_override else ""),
            icon="✅",
        )

# --- Audit trail --------------------------------------------------------------
st.divider()
st.subheader("Recent decisions")
storage_warning()

frame = log.to_frame()
if len(frame):
    recent = frame.tail(12).iloc[::-1]
    st.dataframe(
        recent[
            [
                "timestamp_utc",
                "claim_id",
                "decision_label",
                "decided_by",
                "reason_code",
                "human_overrode_ai",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )
    a1, a2 = st.columns(2)
    a1.metric("Override rate", f"{log.override_rate:.1%}")
    a2.metric("Override-eligible decisions", log.override_eligible_count)
    how_calculated(
        "override rate = human decisions that contradicted a substantive AI "
        "recommendation, over those where the AI made one. Claims merely "
        "routed to a queue for review are excluded — there is nothing to "
        "override when the AI only said 'a human should look at this'."
    )
    st.download_button(
        "Export audit log (CSV)",
        data=log.to_csv(),
        file_name="claimpilot_audit_log.csv",
        mime="text/csv",
    )

footer()
