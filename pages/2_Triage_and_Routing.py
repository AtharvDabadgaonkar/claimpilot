"""Triage and routing: how every claim is scored and where it goes."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from claimpilot import config
from claimpilot.fraud_model import load_model
from claimpilot.pipeline import load_scored
from claimpilot.ui import (
    footer,
    friendly_error,
    how_calculated,
    load_tables,
    page_setup,
)

BAND_COLORS = {"Low": "#2E8B57", "Medium": "#E8A33D", "High": "#C0392B"}
DEST_COLORS = {
    "STP — auto-approved": "#2E8B57",
    "Adjuster queue": "#1F6FEB",
    "Senior adjuster": "#E8A33D",
    "SIU queue": "#C0392B",
}

page_setup(
    "Triage & Routing",
    "Three scores — severity, complexity and fraud signal — each with reason "
    "codes, then a rules-based routing decision.",
)

try:
    tables = load_tables()
    scored = load_scored()
    model = load_model()
except Exception as exc:  # pragma: no cover - UI guard
    friendly_error(exc, "load the scored portfolio")
    st.stop()

claims = tables["claims"]

# --- The hard rule ------------------------------------------------------------
st.success(
    "**ClaimPilot can auto-approve, but it can never auto-deny.** There is no "
    "denial destination in the routing rules, and no way to represent one — "
    "every potential denial goes to a human. A test asserts this.",
    icon="🔒",
)

# --- Routing outcomes ---------------------------------------------------------
st.subheader("Where claims go")

counts = (
    scored["destination_label"].value_counts().rename_axis("Destination").reset_index(name="Claims")
)
k1, k2, k3, k4 = st.columns(4)
k1.metric("STP fast-track", f"{scored['auto_approved'].mean():.1%}")
k2.metric("SIU referral", f"{(scored['destination'] == 'siu_queue').mean():.1%}")
k3.metric("Needs a human", f"{scored['requires_human'].mean():.1%}")
k4.metric("Claims routed", f"{len(scored):,}")
how_calculated(
    "each claim is scored for severity, complexity and fraud, then matched "
    "against config/routing_rules.yaml in order; the first matching rule wins."
)

fig = px.bar(
    counts,
    x="Claims",
    y="Destination",
    orientation="h",
    color="Destination",
    color_discrete_map=DEST_COLORS,
)
fig.update_layout(showlegend=False, height=260, margin={"l": 0, "r": 0, "t": 10, "b": 0})
st.plotly_chart(fig, use_container_width=True)

st.warning(
    f"**An 8% STP rate is low, and that is a property of this dataset, not the "
    f"design.** A real auto carrier would target far more. Here "
    f"{(claims['total_claim_amount'] > 60000).mean():.0%} of claims exceed "
    f"$60,000 and {claims['fraud_reported'].mean():.0%} carry the reported-fraud "
    "flag, so most claims fail the low-severity or low-fraud test. On a "
    "realistic book, both of those shares would be far smaller and the "
    "fast-track pool correspondingly larger.",
    icon="📉",
)

# --- Score distributions ------------------------------------------------------
st.subheader("The three scores")
cols = st.columns(3)
for col, (name, field) in zip(
    cols,
    [("Severity", "severity_band"), ("Complexity", "complexity_band"), ("Fraud signal", "fraud_band")],
    strict=True,
):
    dist = (
        scored[field]
        .value_counts()
        .reindex(["Low", "Medium", "High"])
        .fillna(0)
        .rename_axis("Band")
        .reset_index(name="Claims")
    )
    fig = px.bar(dist, x="Band", y="Claims", color="Band", color_discrete_map=BAND_COLORS)
    fig.update_layout(showlegend=False, height=230, margin={"l": 0, "r": 0, "t": 30, "b": 0}, title=name)
    col.plotly_chart(fig, use_container_width=True)
how_calculated(
    "severity from the claimed amount against bands in config/assumptions.yaml; "
    "complexity from summed points for injuries, parties, missing documents and "
    "coverage questions; fraud from the model probability, escalated one band "
    "when red-flag points reach the configured threshold."
)

# --- Fraud model performance --------------------------------------------------
st.subheader("Fraud model performance")
m = model["metrics"]
f1, f2, f3, f4 = st.columns(4)
f1.metric("ROC AUC (held-out test)", f"{m['roc_auc_test']:.3f}")
f2.metric("ROC AUC (5-fold CV)", f"{m['roc_auc_cv_mean']:.3f}", help=f"±{m['roc_auc_cv_std']:.3f}")
f3.metric(f"Precision @ {m['threshold']}", f"{m['precision_at_threshold']:.3f}")
f4.metric(f"Recall @ {m['threshold']}", f"{m['recall_at_threshold']:.3f}")
how_calculated(
    f"HistGradientBoosting on {m['n_train']} training claims, evaluated on "
    f"{m['n_test']} held-out claims. Threshold {m['threshold']} chosen against "
    "assumed SIU capacity, not F1."
)

st.error(
    f"**Read these with heavy caveats.** The cross-validated AUC "
    f"({m['roc_auc_cv_mean']:.3f}) is lower than the single test-split AUC "
    f"({m['roc_auc_test']:.3f}), so the test figure is a flattering split — "
    f"trust the CV number. The label is a *reported* flag, not an adjudicated "
    f"fraud finding, so the model learns who got referred. And the "
    f"{m['base_rate']:.0%} base rate is roughly 10–25× a real auto portfolio, "
    "which makes precision wildly optimistic.",
    icon="⚠️",
)

with st.expander("Which fields the model is allowed to see"):
    excluded = config.excluded_model_fields()
    extra = config.assumption("fraud_model", "additional_exclusions")
    st.markdown("**Used as model inputs**")
    st.write(", ".join(f"`{f}`" for f in model["features"]))
    st.markdown("**Excluded — protected characteristics and proxies**")
    st.write(", ".join(f"`{f}`" for f in excluded))
    st.markdown("**Excluded — additional proxy risk**")
    for field, why in extra.items():
        st.markdown(f"- `{field}` — {why}")
    st.caption(
        "Exclusion alone does not make a model fair. Outcomes are tested across "
        "these same groups in Phase 4's fairness check."
    )

# --- Claim inspector ----------------------------------------------------------
st.subheader("Inspect a claim")
st.caption("Every score carries reason codes, so an adjuster can disagree with it.")

claim_id = st.selectbox("Claim", scored["claim_id"].tolist(), index=0)
row = scored.loc[scored["claim_id"] == claim_id].iloc[0]
claim = claims.loc[claims["claim_id"] == claim_id].iloc[0]

d1, d2 = st.columns([1, 2])
with d1:
    st.metric("Routed to", row["destination_label"])
    st.caption(f"Rule: **{row['routing_rule']}**")
    st.write(row["routing_reason"])
    st.divider()
    st.write(f"**Claimed:** ${claim['total_claim_amount']:,.0f}")
    st.write(f"**Incident:** {claim['incident_type']} — {claim['incident_severity']}")
    st.write(f"**Coverage:** {row['coverage_status']}")

with d2:
    for label, band_field, reason_field in [
        ("Severity", "severity_band", "severity_reasons"),
        ("Complexity", "complexity_band", "complexity_reasons"),
        ("Fraud signal", "fraud_band", "fraud_reasons"),
    ]:
        band = row[band_field]
        st.markdown(f"**{label}: :{'green' if band == 'Low' else 'orange' if band == 'Medium' else 'red'}[{band}]**")
        for reason in row[reason_field]:
            st.markdown(f"- {reason}")

with st.expander("Coverage reasoning"):
    for reason in row["coverage_reasons"]:
        st.markdown(f"- {reason}")

# --- Rules --------------------------------------------------------------------
with st.expander("The routing rules, in evaluation order"):
    from claimpilot.routing import load_rules

    rules = load_rules()
    st.caption(f"Version `{rules['version']}` — config/routing_rules.yaml")
    for i, rule in enumerate(rules["rules"], 1):
        conditions = ", ".join(
            f"{k} ∈ {v}" for k, v in rule.get("when", {}).items()
        )
        st.markdown(
            f"**{i}. {rule['name']}** → `{rule['destination']}`  \n"
            f"when {conditions}"
        )
    st.markdown(f"**Fallback** → `{rules['fallback']['destination']}`")
    st.caption(
        "Forbidden destinations (the engine refuses to load a rules file "
        "containing any of these): "
        + ", ".join(f"`{d}`" for d in rules["forbidden_destinations"])
    )

footer()
