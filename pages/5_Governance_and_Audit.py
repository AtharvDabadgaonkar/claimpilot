"""Governance: audit trail, fairness testing, model inventory, monitoring."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from claimpilot import config
from claimpilot.audit import DecisionType
from claimpilot.fairness import assess_all
from claimpilot.fairness import summary as fairness_summary
from claimpilot.model_cards import inventory_version, live_metrics_for, load_cards
from claimpilot.monitoring import (
    SignalStatus,
    all_signals,
    band_mix_over_time,
    scores_over_time,
)
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
from claimpilot.workbench import add_deadline_status, demo_today

page_setup(
    "Governance & Audit",
    "Audit trail, fairness testing, model inventory and ongoing monitoring — "
    "designed around the NAIC AI Model Bulletin.",
)

try:
    tables = load_tables()
    scored = load_scored()
    claims = tables["claims"]
    log = get_audit_log()
except Exception as exc:  # pragma: no cover - UI guard
    friendly_error(exc, "load the governance data")
    st.stop()

st.caption(
    "⚠️ *Designed around* the NAIC AI Model Bulletin. This project is not "
    "compliant with, nor certified against, any standard."
)

audit_tab, fairness_tab, inventory_tab, deadlines_tab, monitoring_tab = st.tabs(
    ["Audit trail", "Fairness", "Model inventory", "Deadlines", "Monitoring"]
)

# =============================================================================
# Audit trail
# =============================================================================
with audit_tab:
    st.subheader("Every AI recommendation and human decision")
    frame = log.to_frame()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Records", f"{len(frame):,}")
    k2.metric("Human decisions", f"{(frame['decided_by'] == 'human').sum():,}")
    k3.metric("System auto-approvals", f"{(frame['decided_by'] == 'system').sum():,}")
    k4.metric("Denials", f"{(frame['decision'] == 'deny').sum():,}")
    how_calculated(
        "one record per decision, written at the moment the decision is made. "
        "A denial cannot be recorded without a reason code and a written note."
    )

    st.markdown("**Filter**")
    f1, f2, f3 = st.columns(3)
    decision_filter = f1.multiselect(
        "Decision", [d.value for d in DecisionType], default=[d.value for d in DecisionType]
    )
    decider_filter = f2.multiselect(
        "Decided by", ["human", "system"], default=["human", "system"]
    )
    overrides_only = f3.checkbox("Overrides only")

    view = frame[
        frame["decision"].isin(decision_filter)
        & frame["decided_by"].isin(decider_filter)
    ]
    if overrides_only:
        view = view[view["human_overrode_ai"]]

    search = st.text_input("Claim ID contains", placeholder="CLM-2015-")
    if search:
        view = view[view["claim_id"].str.contains(search, case=False, na=False)]

    st.dataframe(
        view[
            [
                "timestamp_utc", "claim_id", "decision_label", "decided_by",
                "decider_id", "reason_code", "ai_recommendation",
                "human_overrode_ai", "note",
            ]
        ].iloc[::-1],
        hide_index=True,
        use_container_width=True,
        height=360,
    )
    st.caption(f"Showing {len(view):,} of {len(frame):,} records.")

    with st.expander("What each record captures"):
        st.markdown(
            """
- **When and which claim** — timestamp and claim ID
- **What the AI said** — recommendation, severity / complexity / fraud bands,
  fraud probability, suggested reserve
- **Which versions produced it** — triage, routing rules, fraud model, reserve model
- **What inputs were used**
- **What a human did** — decision, who decided, reason code, note
- **Whether the human overrode the AI**
"""
        )

    st.download_button(
        "⬇️ Export full audit trail (CSV)",
        data=log.to_csv(),
        file_name="claimpilot_audit_log.csv",
        mime="text/csv",
    )
    storage_warning()

# =============================================================================
# Fairness
# =============================================================================
with fairness_tab:
    st.subheader("Are outcomes evenly distributed?")
    results = assess_all(scored, claims)
    report = fairness_summary(results)

    st.info(
        "**Protected attributes are never model inputs — but that is not "
        "enough.** Other features can correlate with them, so outcomes are "
        "measured across the same groups the models are forbidden to see.",
        icon="⚖️",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Tests run", report["tests_run"])
    c2.metric("Flagged", report["tests_flagged"])
    c3.metric("Worst ratio", f"{report['worst_ratio']:.3f}" if report["worst_ratio"] else "—")
    how_calculated(
        "disparate impact ratio = lowest group's rate ÷ highest group's rate, "
        f"flagged below {report['threshold']}. Groups under "
        f"{config.assumption('fairness', 'min_group_size')} claims are shown "
        "but excluded from the ratio."
    )

    if report["tests_flagged"]:
        st.error(
            f"**{report['tests_flagged']} of {report['tests_assessable']} tests fall below "
            f"{report['threshold']}.** This portfolio does not pass its own "
            "fairness check, and the demo reports that rather than hiding it.",
            icon="🚩",
        )

    rows = []
    for result in results:
        rows.append(
            {
                "Attribute": result.attribute,
                "Outcome": result.outcome,
                "Ratio": result.disparate_impact_ratio,
                "Lowest group": result.lowest_group,
                "Highest group": result.highest_group,
                "Verdict": result.verdict,
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
        column_config={"Ratio": st.column_config.NumberColumn(format="%.3f")},
    )

    st.markdown("#### Group rates")
    chosen = st.selectbox(
        "Show",
        range(len(results)),
        format_func=lambda i: f"{results[i].outcome} by {results[i].attribute}",
    )
    result = results[chosen]
    group_frame = pd.DataFrame([g.to_dict() for g in result.groups])
    fig = px.bar(
        group_frame, x="group", y="rate", color="too_small",
        color_discrete_map={True: "#9AA5B1", False: "#1F6FEB"},
        labels={"group": result.attribute, "rate": result.outcome, "too_small": "Too small to assess"},
    )
    fig.add_hline(
        y=group_frame.loc[~group_frame["too_small"], "rate"].max() * result.threshold,
        line_dash="dash", line_color="#C0392B",
        annotation_text=f"{result.threshold:.0%} of the highest group",
    )
    fig.update_layout(height=320, margin={"l": 0, "r": 0, "t": 20, "b": 0})
    st.plotly_chart(fig, use_container_width=True)
    if result.excluded_small_groups:
        st.caption(
            "Excluded from the ratio (too few claims): "
            + ", ".join(result.excluded_small_groups)
        )

    with st.expander("⚠️ Why this portfolio fails, and what the 0.8 threshold is"):
        st.markdown(
            """
**The 0.8 threshold is borrowed, not an insurance standard.** It is the EEOC's
four-fifths rule from US *employment* law, used here because it is widely
understood. Insurance discrimination is governed by state unfair trade
practice and unfair claims settlement acts, which define no single numeric
test. A real carrier would agree its metrics with counsel and its regulators.

**Why the disparities appear even though the attributes are excluded.** Age is
not a model input, yet the oldest band is fast-tracked at roughly a third the
rate of the highest band. The path runs through correlation: that band has the
fewest low-severity claims and the highest reported-fraud rate in this data, so
it fails the low-severity and low-fraud gates more often. The fraud label is
itself a *reported* flag, so any historical bias in who got referred is
reproduced here.

**This is the point of outcome testing.** Excluding a protected attribute did
not prevent a disparate outcome. On a real portfolio this result would trigger
investigation before deployment, not a note in a demo.
"""
        )
        st.caption(report["threshold_note"])

# =============================================================================
# Model inventory
# =============================================================================
with inventory_tab:
    st.subheader("Model inventory")
    cards = load_cards()
    st.caption(f"Inventory `{inventory_version()}` — config/model_cards.yaml")

    tiers = pd.DataFrame(
        [{"Component": c.name, "Risk tier": c.risk_tier.label, "Version": c.version,
          "Owner": c.owner, "Last evaluated": c.last_evaluated} for c in cards]
    )
    st.dataframe(tiers, hide_index=True, use_container_width=True)
    how_calculated(
        "risk tier reflects how much a component can change what happens to a "
        "claim: low = advisory, medium = shapes routing or valuation under "
        "review, high = can change the claim's path."
    )

    for card in cards:
        with st.expander(f"{card.risk_tier.icon} {card.name} — {card.risk_tier.label} risk"):
            st.markdown(f"**Purpose.** {card.purpose}")
            st.markdown(f"**Technique.** {card.technique}")
            st.markdown(f"**Owner.** {card.owner} · **Last evaluated.** {card.last_evaluated}")
            st.markdown(f"**Oversight.** {card.human_oversight}")

            metrics = live_metrics_for(card.id)
            if metrics:
                st.markdown("**Current measured performance**")
                cols = st.columns(len(metrics))
                for col, (label, value) in zip(cols, metrics.items(), strict=True):
                    col.metric(label, value)
                st.caption(
                    "Read live from reports/metrics.json, so a card cannot "
                    "drift from the model it describes."
                )

            st.markdown("**Training data.** " + card.training_data)
            st.markdown("**Inputs**")
            for item in card.inputs:
                st.markdown(f"- {item}")
            st.markdown("**Excluded inputs**")
            for item in card.excluded_inputs:
                st.markdown(f"- {item}")
            st.markdown("**Limitations**")
            for item in card.limitations:
                st.markdown(f"- {item}")
            st.markdown("**Failure modes**")
            for item in card.failure_modes:
                st.markdown(f"- {item}")

# =============================================================================
# Deadlines
# =============================================================================
with deadlines_tab:
    st.subheader("Claim deadline tracker")
    status = add_deadline_status(tables["deadlines"])

    st.warning(
        "**Illustrative deadlines only.** Real acknowledge / decide / pay "
        "clocks are set by state unfair claims settlement practice rules and "
        "vary by jurisdiction and claim type. These are placeholders.",
        icon="📅",
    )
    st.caption(f"🗓️ Demo date: **{demo_today()}**")

    d1, d2, d3 = st.columns(3)
    d1.metric("Breached", f"{(status['deadline_status'] == 'Breached').sum():,}")
    d2.metric("At risk", f"{(status['deadline_status'] == 'At risk').sum():,}")
    d3.metric("On track", f"{(status['deadline_status'] == 'On track').sum():,}")
    how_calculated(
        "status follows the next OPEN obligation. Digital FNOL acknowledges on "
        "submission, so the acknowledge clock is met and does not count as a "
        "pending breach."
    )

    mix = (
        status["deadline_status"].value_counts().rename_axis("Status").reset_index(name="Claims")
    )
    fig = px.bar(
        mix, x="Status", y="Claims", color="Status",
        color_discrete_map={"Breached": "#C0392B", "At risk": "#E8A33D", "On track": "#2E8B57"},
    )
    fig.update_layout(showlegend=False, height=280, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"The high breach rate is mostly an artifact: the dataset's losses run "
        f"Jan–Mar 2015 and the demo views them from a fixed date of "
        f"{demo_today()}, so older claims have genuinely blown the 30-day "
        f"decide clock. The acknowledge deadline is met on "
        f"{(status['acknowledge_status'] == 'Met').mean():.0%} of claims.",
        icon="ℹ️",
    )

    st.dataframe(
        status[
            ["claim_id", "reported_date", "decide_due", "days_to_decide_due",
             "deadline_status", "acknowledge_status", "pay_status"]
        ].sort_values("days_to_decide_due").head(50),
        hide_index=True,
        use_container_width=True,
    )

# =============================================================================
# Monitoring
# =============================================================================
with monitoring_tab:
    st.subheader("Ongoing monitoring")
    signals = all_signals(log, scored, claims)

    for signal in signals:
        box = {
            SignalStatus.OK: st.success,
            SignalStatus.WARN: st.warning,
            SignalStatus.ALERT: st.error,
            SignalStatus.INSUFFICIENT: st.info,
        }[signal.status]
        value = f"{signal.value:.3f}" if signal.value is not None else "—"
        box(f"{signal.status.icon} **{signal.name}: {value}** — {signal.detail}")

    st.markdown("#### Score distributions over time")
    trend = scores_over_time(scored, claims)
    fig = px.line(
        trend, x="period",
        y=["mean_fraud_probability", "stp_rate", "siu_rate"],
        labels={"period": "Week of loss", "value": "Rate", "variable": "Measure"},
        markers=True,
    )
    fig.update_layout(height=320, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig, use_container_width=True)
    how_calculated(
        "weekly means over the date of loss. Weeks are thin (~110 claims each), "
        "so movement here is noise as much as signal."
    )

    mix = band_mix_over_time(scored, claims)
    fig = px.area(
        mix, x="period", y="share", color="fraud_band",
        color_discrete_map={"Low": "#2E8B57", "Medium": "#E8A33D", "High": "#C0392B"},
        labels={"period": "Week of loss", "share": "Share of claims", "fraud_band": "Fraud band"},
    )
    fig.update_layout(height=300, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Why a near-zero override rate would also be a warning"):
        st.markdown(
            """
The override rate is monitored against a **floor as well as a ceiling**. A high
rate questions the model. But a rate near zero is not a success: if humans
never disagree with the AI, either the model is perfect or nobody is really
reviewing. The second is far more likely, and a review process that always
agrees is not meaningful oversight.
"""
        )
        st.caption(
            " ".join(str(config.assumption("monitoring", "drift_note")).split())
        )

footer()
