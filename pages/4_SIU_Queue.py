"""SIU queue: claims referred for investigation, and why."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from claimpilot.audit import (
    DecidedBy,
    DecisionType,
    MissingReasonCodeError,
    explain_validation_error,
    reason_codes_for,
)
from claimpilot.fraud_model import MODEL_VERSION as FRAUD_VERSION
from claimpilot.fraud_model import load_model
from claimpilot.pipeline import load_scored
from claimpilot.ui import (
    footer,
    friendly_error,
    get_audit_log,
    how_calculated,
    load_tables,
    page_setup,
    storage_warning,
)
from claimpilot.workbench import build_queue, demo_today

page_setup(
    "SIU Queue",
    "Claims referred to the Special Investigation Unit, with the red flags "
    "that triggered each referral.",
)

try:
    tables = load_tables()
    scored = load_scored()
    queue = build_queue(scored, tables["claims"], tables["deadlines"], ["siu_queue"])
    log = get_audit_log()
    model = load_model()
except Exception as exc:  # pragma: no cover - UI guard
    friendly_error(exc, "build the SIU queue")
    st.stop()

st.caption(f"🗓️ Demo date: **{demo_today()}**")

st.info(
    "**A referral is not an accusation.** SIU investigates; it does not deny. "
    "ClaimPilot cannot auto-deny a claim, and a fraud score alone is never a "
    "basis for refusing payment.",
    icon="⚖️",
)

m = model["metrics"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Claims referred", f"{len(queue):,}")
k2.metric("Referral rate", f"{len(queue) / len(scored):.1%}")
k3.metric(f"Model precision @ {m['threshold']}", f"{m['precision_at_threshold']:.2f}")
k4.metric(f"Model recall @ {m['threshold']}", f"{m['recall_at_threshold']:.2f}")
how_calculated(
    "a claim is referred when the fraud band is High — either the model "
    "probability exceeds the configured band, or red-flag points escalate it. "
    "The threshold is set against assumed SIU capacity, not F1."
)

st.warning(
    f"**Expect roughly {1 - m['precision_at_threshold']:.0%} of these referrals to be "
    f"unfounded.** At a precision of {m['precision_at_threshold']:.2f}, most of an "
    "investigator's queue is still legitimate claims. That is the cost of "
    f"catching {m['recall_at_threshold']:.0%} of the flagged ones, and it is why "
    "the referral rate is capped by capacity rather than maximised. The label "
    "is also a *reported* flag, not an adjudicated finding, so the model has "
    "learned who got referred historically — including any bias in that "
    "process.",
    icon="⚠️",
)

# --- Red flag frequency -------------------------------------------------------
st.subheader("What is triggering referrals")

flag_labels = {
    "FRD_LATE_NIGHT": "Loss outside normal hours",
    "FRD_NO_REPORT_HIGH_VALUE": "No police report on a high-value claim",
    "FRD_SHORT_TENURE": "Policy only recently taken out",
    "FRD_NO_CORROBORATION": "No witnesses and no police report",
    "FRD_CLAIM_PREMIUM_RATIO": "Claim is a large multiple of the premium",
    "FRD_FLAG_ESCALATION": "Red flags escalated the band",
    "FRD_MODEL_SCORE": "Model probability",
}
counts: dict[str, int] = {}
for codes in queue["fraud_codes"]:
    for code in codes:
        if code == "FRD_MODEL_SCORE":
            continue
        counts[flag_labels.get(code, code)] = counts.get(flag_labels.get(code, code), 0) + 1

if counts:
    flags = (
        pd.DataFrame({"Red flag": list(counts), "Claims": list(counts.values())})
        .sort_values("Claims", ascending=True)
    )
    fig = px.bar(flags, x="Claims", y="Red flag", orientation="h")
    fig.update_traces(marker_color="#C0392B")
    fig.update_layout(height=300, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig, use_container_width=True)
    how_calculated(
        "count of referred claims carrying each red-flag reason code. A claim "
        "can carry several. The model probability is excluded from this chart "
        "because every claim has one."
    )

# --- Queue --------------------------------------------------------------------
st.subheader("Referred claims")
st.dataframe(
    queue[
        [
            "claim_id",
            "priority_score",
            "fraud_probability",
            "total_claim_amount",
            "incident_type",
            "coverage_status",
            "deadline_status",
        ]
    ].head(50),
    hide_index=True,
    use_container_width=True,
    column_config={
        "claim_id": "Claim",
        "priority_score": st.column_config.NumberColumn("Priority", format="%.1f"),
        "fraud_probability": st.column_config.NumberColumn("Fraud prob.", format="%.2f"),
        "total_claim_amount": st.column_config.NumberColumn("Claimed", format="$%d"),
        "incident_type": "Incident",
        "coverage_status": "Coverage",
        "deadline_status": "Deadline",
    },
)

if queue.empty:
    st.info("No claims currently referred to SIU.", icon="ℹ️")
    footer()
    st.stop()

# --- Investigate --------------------------------------------------------------
st.divider()
st.subheader("Investigate a claim")

claim_id = st.selectbox("Claim", queue["claim_id"].head(50).tolist())
row = queue.loc[queue["claim_id"] == claim_id].iloc[0]

left, right = st.columns([2, 1])
with left:
    st.markdown("**Why this claim was referred**")
    for reason in row["fraud_reasons"]:
        st.markdown(f"- {reason}")
    st.markdown("**Complexity drivers**")
    for reason in row["complexity_reasons"]:
        st.markdown(f"- {reason}")
with right:
    st.metric("Fraud probability", f"{row['fraud_probability']:.0%}")
    st.metric("Claimed", f"${row['total_claim_amount']:,.0f}")
    st.write(f"**Incident:** {row['incident_type']}")
    st.write(f"**Coverage:** {row['coverage_status']}")
    st.write(f"**Deadline:** {row['deadline_status']}")

st.markdown("#### Investigation outcome")
st.caption(
    "SIU can clear a claim back to the adjuster or request more information. "
    "Only a human adjuster can deny, and only with a reason code and a note."
)

outcome = st.radio(
    "Outcome",
    [DecisionType.APPROVE, DecisionType.REQUEST_INFO, DecisionType.DENY],
    format_func=lambda d: {
        DecisionType.APPROVE: "Clear — no fraud indicators substantiated",
        DecisionType.REQUEST_INFO: "Request further information",
        DecisionType.DENY: "Refer for denial (human decision)",
    }[d],
)

codes = reason_codes_for(outcome)
reason_code = st.selectbox(
    "Reason code (required)", list(codes), format_func=lambda c: f"{c} — {codes[c]}"
) if codes else None

note = st.text_area(
    "Investigator note" + (" (required, min 20 characters)" if outcome is DecisionType.DENY else ""),
    placeholder="Record the findings…",
)

if st.button("Record outcome", type="primary"):
    try:
        record = log.record_decision(
            claim_id=claim_id,
            ai_recommendation=row["destination"],
            ai_severity_band=row["severity_band"],
            ai_complexity_band=row["complexity_band"],
            ai_fraud_band=row["fraud_band"],
            ai_fraud_probability=float(row["fraud_probability"]),
            triage_version=row["triage_version"],
            rules_version=row["rules_version"],
            fraud_model_version=FRAUD_VERSION,
            inputs_used=["fraud model", "red flags", "claim record"],
            decision=outcome,
            decided_by=DecidedBy.HUMAN,
            decider_id="siu-investigator-demo",
            reason_code=reason_code,
            note=note or None,
        )
    except (MissingReasonCodeError, ValueError) as exc:
        st.error(
            f"That outcome can't be recorded. {explain_validation_error(exc)}",
            icon="🚫",
        )
    else:
        st.success(f"Recorded as `{record.record_id}`.", icon="✅")

storage_warning()
footer()
