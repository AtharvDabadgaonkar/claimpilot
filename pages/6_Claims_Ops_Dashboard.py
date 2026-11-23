"""Claims Ops Dashboard: KPI tiles, guardrails and the backtest."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from claimpilot import config
from claimpilot.backtest import (
    by_route,
    comparison_frame,
    deltas,
    lae_breakeven,
    load_backtest,
    siu_hit_rate,
    summary,
)
from claimpilot.kpis import load_metrics
from claimpilot.ui import (
    footer,
    friendly_error,
    how_calculated,
    page_setup,
    simulated_badge,
)

page_setup(
    "Claims Ops Dashboard",
    "Operating KPIs, guardrails, and a paired backtest against an "
    "all-manual baseline.",
)

try:
    bt = load_backtest()
    metrics = load_metrics()
except Exception as exc:  # pragma: no cover - UI guard
    friendly_error(exc, "load the backtest and metrics")
    st.stop()

s = summary(bt)
d = deltas(bt)
ops = metrics["ops_dashboard"]
kpi = ops["kpis"]
guard = ops["guardrails"]
manual = s["arms"]["manual"]
cp = s["arms"]["claimpilot"]

st.error(
    "**These deltas are not evidence.** The two arms are *paired*: the "
    "ClaimPilot arm is derived from each claim's own baseline draw by applying "
    "the multipliers in `config/assumptions.yaml`. That removes sampling noise, "
    "but it also means every difference below is an **arithmetic consequence "
    "of the assumptions**, not a measured outcome. Only the routing mix and the "
    "SIU hit rate are measured. A real backtest replays decisions against "
    "observed outcomes; the public dataset has none.",
    icon="⚠️",
)

# ---------------------------------------------------------------------------
# KPI tiles
# ---------------------------------------------------------------------------
st.subheader("Operating KPIs")
simulated_badge(
    "Cycle time, touch time, cost and leakage are simulated. STP rate, SIU hit "
    "rate and reserve accuracy are measured."
)

r1 = st.columns(4)
r1[0].metric("Straight-through processing", f"{kpi['stp_rate']:.1%}")
r1[1].metric(
    "Mean cycle time",
    f"{kpi['mean_cycle_days']:.1f} d",
    delta=f"{d['cycle_days_pct']:+.1%} vs manual",
    delta_color="inverse",
)
r1[2].metric(
    "Time to first contact",
    f"{kpi['time_to_first_contact_hours']:.0f} h",
    delta=f"{d['first_contact_pct']:+.0%} vs manual",
    delta_color="inverse",
)
r1[3].metric(
    "LAE per claim",
    f"${kpi['lae_per_claim_usd']:,.0f}",
    delta=f"{d['lae_per_claim_pct']:+.1%} vs manual",
    delta_color="inverse",
)

r2 = st.columns(4)
r2[0].metric("Reserve accuracy (MAE)", f"${kpi['reserve_mae_usd']:,.0f}")
r2[1].metric(
    "80% interval coverage",
    f"{kpi['reserve_interval_coverage']:.1%}",
    help="Share of claims whose actual amount fell inside the P10–P90 band. "
    "Below 80% means the interval is too narrow.",
)
r2[2].metric(
    "SIU referral hit rate",
    f"{kpi['siu_hit_rate']:.1%}",
    delta=f"base rate {kpi['siu_base_rate']:.1%}",
    delta_color="off",
)
r2[3].metric(
    "Adjuster FTE-months",
    f"{kpi['adjuster_fte_months']:.0f}",
    delta=f"{d['adjuster_fte_months']:+.0f} vs manual",
    delta_color="inverse",
)
how_calculated(
    "STP rate from routing decisions in scored.parquet; cycle time, touch "
    "time and LAE from backtest.parquet (paired arms); reserve accuracy from "
    "held-out model evaluation; SIU hit rate = share of referred claims "
    "carrying the reported-fraud flag; FTE-months = total touch hours ÷ "
    f"{config.assumption('backtest', 'adjuster_hours_per_month')} hours."
)

st.info(
    f"**Automation did not reduce expense here, and the dashboard says so.** "
    f"LAE per claim rose {d['lae_per_claim_pct']:+.1%} and mean cycle time rose "
    f"{d['cycle_days_pct']:+.1%}, because {s['referred_to_siu']} claims "
    f"({s['referred_to_siu'] / s['claims']:.0%}) were referred to SIU and each "
    f"referral adds investigation time the all-manual arm never spends. Total "
    f"cost still falls {-d['total_cost_pct']:.1%} once estimated leakage is "
    "counted. See the break-even section below.",
    icon="📉",
)

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
st.subheader("Guardrails")
st.caption(
    "Efficiency numbers mean nothing without these beside them. Two are "
    "currently failing their own thresholds."
)

g1 = st.columns(5)
g1[0].metric(
    "Estimated leakage",
    f"${guard['estimated_leakage_total_usd'] / 1e6:.2f}M",
    delta=f"{d['leakage_total_usd'] / 1e6:+.2f}M vs manual",
    delta_color="inverse",
)
g1[1].metric("Reopen rate", f"{guard['reopen_rate']:.1%}")
g1[2].metric(
    "Worst fairness ratio",
    f"{guard['worst_fairness_ratio']:.3f}",
    delta=f"{guard['fairness_tests_flagged']}/{guard['fairness_tests_assessable']} flagged",
    delta_color="inverse",
)
g1[3].metric("Override rate", f"{guard['override_rate']:.1%}")
g1[4].metric("Claims auto-denied", guard["auto_denied_count"])

g2 = st.columns(3)
g2[0].metric("Deadline breaches", f"{guard['deadline_breached_rate']:.1%}")
g2[1].metric("Monitoring alerts", guard["monitoring_alerts"])
g2[2].metric("Monitoring warnings", guard["monitoring_warnings"])
how_calculated(
    "Every guardrail is read from reports/metrics.json, not recomputed here, "
    "so a tile cannot disagree with the section it came from. Leakage is a "
    "modelled proxy; fairness is a disparate-impact ratio against a 0.80 "
    "threshold; override rate counts only substantive AI recommendations."
)

st.warning(
    f"**Two guardrails are red.** The fairness check flags "
    f"{guard['fairness_tests_flagged']} of "
    f"{guard['fairness_tests_assessable']} tests (worst ratio "
    f"{guard['worst_fairness_ratio']:.3f} against a 0.80 threshold), and "
    f"{guard['deadline_breached_rate']:.0%} of claims sit past an illustrative "
    "deadline. Neither is hidden, and neither threshold was moved to make the "
    "dashboard look better. Details on the Governance & Audit page.",
    icon="🚩",
)
st.success(
    f"**{guard['auto_denied_count']} claims auto-denied, by construction.** The "
    "routing engine has no denial destination to reach. Every potential denial "
    "goes to a human.",
    icon="🛡️",
)

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
st.subheader("Backtest: ClaimPilot vs. all manual")

tab_arms, tab_route, tab_break, tab_siu = st.tabs(
    ["Arms side by side", "Where it comes from", "Break-even", "SIU referrals"]
)

with tab_arms:
    comp = comparison_frame(bt)
    st.dataframe(
        comp,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Mean LAE ($)": st.column_config.NumberColumn(format="$%.0f"),
            "Est. leakage, total ($)": st.column_config.NumberColumn(format="$%.0f"),
            "LAE + leakage, total ($)": st.column_config.NumberColumn(format="$%.0f"),
            "Reopen rate": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    how_calculated(
        "Both rows cover the same 1,000 claims. The manual row uses the "
        "lognormal draws made in Phase 1; the ClaimPilot row applies this "
        "claim's route multipliers to that same draw."
    )

    cost = pd.DataFrame(
        {
            "Arm": ["All manual", "All manual", "ClaimPilot", "ClaimPilot"],
            "Component": ["Expense (LAE)", "Estimated leakage"] * 2,
            "USD": [
                manual["total_lae_usd"],
                manual["total_leakage_usd"],
                cp["total_lae_usd"],
                cp["total_leakage_usd"],
            ],
        }
    )
    fig = px.bar(
        cost,
        x="Arm",
        y="USD",
        color="Component",
        barmode="stack",
        title="Total cost of handling: expense plus estimated leakage",
        color_discrete_sequence=["#4C78A8", "#E45756"],
    )
    fig.update_layout(yaxis_title="USD (total, 1,000 claims)", height=380)
    st.plotly_chart(fig, use_container_width=True)
    how_calculated(
        "Expense is the sum of simulated adjuster and SIU time plus fixed "
        "per-claim cost. Leakage is a modelled proxy: a share of the paid "
        "amount on suspicious claims the arm failed to catch. Read the two "
        "together — the manual arm buys low expense by never investigating."
    )

with tab_route:
    st.markdown(
        "The portfolio average hides the mechanism. Split by route, the "
        "picture is unambiguous: fast-tracked claims collapse to almost "
        "nothing, assisted claims get modestly cheaper, and SIU referrals "
        "cost **more** than the manual arm would have spent — because the "
        "manual arm investigates nothing at all."
    )
    routes = by_route(bt)
    st.dataframe(routes, hide_index=True, use_container_width=True)
    fig = px.bar(
        routes,
        x="destination",
        y="LAE delta ($)",
        color="LAE delta ($)",
        color_continuous_scale=["#2E7D32", "#F5F5F5", "#C62828"],
        color_continuous_midpoint=0,
        title="LAE per claim: ClaimPilot minus all-manual, by route",
    )
    fig.update_layout(height=380, showlegend=False, xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    how_calculated(
        "Mean LAE per claim in each arm, grouped by the destination routing "
        "actually chose, then differenced. Positive bars cost more than manual."
    )

with tab_break:
    be = lae_breakeven(bt)
    st.markdown(
        "If ClaimPilot's expense line is higher, how much of that is the "
        "**investigation programme** rather than the automation? SIU time is "
        "the only term that scales with the referral count, so the question "
        "solves exactly."
    )
    b1, b2, b3 = st.columns(3)
    b1.metric("Automation headroom", f"${be['automation_headroom_usd']:,.0f}")
    b2.metric("Break-even referral rate", f"{be['breakeven_referral_rate']:.1%}")
    b3.metric(
        "Actual referral rate",
        f"{be['actual_referral_rate']:.1%}",
        delta=f"{be['actual_referral_rate'] - be['breakeven_referral_rate']:+.1%}",
        delta_color="inverse",
    )
    st.markdown(
        f"The automation saves **\\${be['automation_headroom_usd']:,.0f}** against "
        f"the manual arm before any investigation spend. Each referral costs "
        f"**\\${be['cost_per_referral_usd']:,.0f}**, so the book can carry "
        f"**{be['breakeven_referrals']} referrals "
        f"({be['breakeven_referral_rate']:.1%})** before automation stops paying "
        f"for the SIU programme on expense alone. This dataset forces "
        f"**{be['actual_referrals']} ({be['actual_referral_rate']:.1%})**, so "
        f"{be['verdict']}."
    )
    st.info(
        "**This is a property of the dataset, not the design.** The sample's "
        "reported-fraud rate is 24.7% — an order of magnitude above a real "
        "auto portfolio. At a realistic flag rate the referral volume, and "
        "therefore the investigation cost, would sit well under the break-even "
        "line. That recalibration is exactly what a real deployment would "
        "have to do before quoting any saving.",
        icon="🧪",
    )
    how_calculated(
        "Break-even count = (manual LAE total − ClaimPilot LAE total "
        "excluding SIU time) ÷ cost per referral. Exact arithmetic on the "
        "assumed rates, not a simulation."
    )

with tab_siu:
    hits = siu_hit_rate(bt)
    s1, s2, s3 = st.columns(3)
    s1.metric("Referred to SIU", f"{hits['referred']:,}")
    s2.metric(
        "Hit rate (precision)",
        f"{hits['hit_rate_precision']:.1%}",
        delta=f"{hits['hit_rate_precision'] - hits['portfolio_flag_rate']:+.1%} vs base rate",
    )
    s3.metric("Flagged claims caught", f"{hits['flagged_caught_recall']:.1%}")
    st.markdown(
        f"{hits['hits']:,} of {hits['referred']:,} referrals carry the "
        f"reported-fraud flag — **{hits['hit_rate_precision']:.1%}** against a "
        f"portfolio base rate of {hits['portfolio_flag_rate']:.1%}. "
        f"{hits['flagged_not_referred']:,} flagged claims were **not** "
        "referred, and the leakage estimate charges ClaimPilot for every one."
    )
    st.warning(f"**Read with care.** {hits['label_caveat']}", icon="🏷️")
    how_calculated(
        "Precision = referred claims carrying the reported-fraud flag ÷ all "
        "referred claims. Recall = the same numerator ÷ all flagged claims. "
        "Measured from routing decisions, not assumed."
    )

with st.expander("Every assumption behind these numbers"):
    st.markdown(
        "Nothing on this page is hard-coded in the app. The multipliers, "
        "hourly rates, leakage rates and FTE hours all live in "
        "`config/assumptions.yaml`, and `make backtest && make metrics` "
        "regenerates every figure from them."
    )
    st.markdown("**Interpretation note from the config:**")
    st.info(s["interpretation_note"])
    st.markdown("**Why the two arms' fraud-miss rates are not comparable:**")
    st.warning(str(config.assumption("leakage", "claimpilot_rate_source")).strip())
    st.json(
        {
            "handling_time_hours": config.assumption("handling_time_hours"),
            "cycle_time_days": config.assumption("cycle_time_days"),
            "costs": config.assumption("costs"),
            "leakage": config.assumption("leakage"),
            "backtest": config.assumption("backtest"),
        },
        expanded=False,
    )

footer()
