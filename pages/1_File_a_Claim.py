"""Digital FNOL intake: submit a claim, watch it get extracted and checked."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from claimpilot.coverage import CoverageStatus, check_coverage
from claimpilot.intake import FNOLSubmission, LossType, extract_fnol
from claimpilot.llm.masking import mask_text
from claimpilot.ui import (
    footer,
    friendly_error,
    load_tables,
    page_setup,
    simulated_badge,
)

page_setup(
    "File a Claim (FNOL)",
    "Digital first notice of loss. The free-text account is turned into "
    "structured fields and checked against the policy.",
)

try:
    tables = load_tables()
except Exception as exc:  # pragma: no cover - UI guard
    friendly_error(exc, "load the claims data")
    st.stop()

claims = tables["claims"]
policies = tables["policies"]
narratives = tables["narratives"]

st.info(
    "**Demo mode: rule-based AI.** Extraction here is deterministic rules and "
    "regular expressions, not a language model — so it needs no API key and "
    "cannot invent a value. See the caveat at the bottom of this page.",
    icon="🤖",
)

# --- Pick a sample narrative or write your own --------------------------------
left, right = st.columns([2, 1])

with right:
    st.subheader("Start from a sample")
    simulated_badge("Sample narratives are simulated")
    sample_ids = narratives["claim_id"].head(200).tolist()
    chosen = st.selectbox("Sample claim", ["(write my own)"] + sample_ids, index=1)

    if chosen != "(write my own)":
        row = narratives.loc[narratives["claim_id"] == chosen].iloc[0]
        default_text = row["narrative"]
        claim_row = claims.loc[claims["claim_id"] == chosen].iloc[0]
        policy_row = policies.loc[policies["claim_id"] == chosen].iloc[0]
        default_policy = policy_row["mock_policy_number"]
    else:
        default_text = ""
        claim_row = None
        policy_row = None
        default_policy = "NWM-000000"

with left:
    st.subheader("Tell us what happened")
    with st.form("fnol"):
        policy_number = st.text_input("Policy number (mock)", value=default_policy)
        c1, c2 = st.columns(2)
        loss_date = c1.date_input("Date of loss", value=date(2015, 2, 1))
        reported_date = c2.date_input("Date reported", value=date(2015, 2, 3))
        loss_type_choice = st.selectbox(
            "Type of loss", [t.value for t in LossType if t is not LossType.UNKNOWN]
        )
        c3, c4 = st.columns([3, 1])
        loss_city = c3.text_input("City", value="Columbus")
        loss_state = c4.text_input("State", value="OH", max_chars=2)
        description = st.text_area(
            "In your own words, what happened?",
            value=default_text,
            height=180,
            placeholder="Describe the incident…",
        )
        photo = st.file_uploader(
            "Photo of the damage (optional)", type=["png", "jpg", "jpeg"]
        )
        submitted = st.form_submit_button("Submit claim", type="primary")

st.caption(
    "ℹ️ Photos are stored with the claim but are **not analysed** in this MVP. "
    "Photo-based damage estimation is listed as Phase 2 work in the PRD."
)

if not submitted:
    footer()
    st.stop()

# --- Validate the submission --------------------------------------------------
try:
    submission = FNOLSubmission(
        policy_number=policy_number,
        reported_date=reported_date,
        loss_date=loss_date,
        loss_type=LossType(loss_type_choice),
        loss_city=loss_city,
        loss_state=loss_state,
        description=description,
        photo_filename=photo.name if photo else None,
    )
except Exception as exc:
    st.error(f"That submission isn't valid: {exc}", icon="⚠️")
    footer()
    st.stop()

st.success(f"Claim received for policy {submission.policy_number}.", icon="✅")

# --- Extraction ---------------------------------------------------------------
st.subheader("1. What the AI extracted")
extracted = extract_fnol(submission.description)

e1, e2, e3 = st.columns(3)
e1.metric("Fields populated", f"{extracted.fields_found}/{extracted.fields_attempted}")
e2.metric("Completeness", f"{extracted.completeness:.0%}")
e3.metric("Still missing", len(extracted.missing_fields()))
st.caption(
    "ℹ️ How it's calculated: count of fields the rule-based extractor populated, "
    "out of the 15 it attempts. A field is left empty rather than guessed."
)

display = {
    "Date of loss": extracted.loss_date,
    "Loss type": extracted.loss_type.value,
    "Collision type": extracted.collision_type.value if extracted.collision_type else None,
    "Damage areas": ", ".join(a.value for a in extracted.damage_areas) or None,
    "Injury reported": extracted.injury_reported,
    "Number injured": extracted.num_injured,
    "Police report": extracted.police_report_available,
    "Vehicles involved": extracted.num_vehicles,
    "Witnesses": extracted.witnesses,
    "Location": f"{extracted.city}, {extracted.state}" if extracted.city else None,
    "Vehicle": f"{extracted.auto_year} {extracted.auto_make}" if extracted.auto_make else None,
    "Estimated amount": (
        f"${extracted.estimated_amount_usd:,.0f}"
        if extracted.estimated_amount_usd is not None
        else None
    ),
}
frame = pd.DataFrame(
    [
        {"Field": k, "Extracted": "— not stated —" if v is None else str(v)}
        for k, v in display.items()
    ]
)
st.dataframe(frame, hide_index=True, use_container_width=True)

if extracted.missing_fields():
    st.warning(
        "**Not stated in the description:** "
        + ", ".join(extracted.missing_fields())
        + ". These are left empty rather than assumed — an unanswered question is "
        "a documentation gap, not a 'no'.",
        icon="📋",
    )

# --- PII masking --------------------------------------------------------------
st.subheader("2. Personal data removed before any AI call")
masked = mask_text(submission.description)
if masked.masked_count:
    st.write(
        f"Masked **{masked.masked_count}** item(s): "
        + ", ".join(f"`{k}` ×{v}" for k, v in masked.kinds.items())
    )
    with st.expander("See exactly what an LLM would receive"):
        st.code(masked.text, language=None)
else:
    st.write("No personal data detected in this description.")
st.caption(
    "ℹ️ How it's calculated: regex-based detection of names, emails, phone "
    "numbers, policy numbers, addresses, VINs and SSNs, applied before any text "
    "leaves the process. Demo grade — a real deployment would use a validated "
    "PII service."
)

# --- Coverage -----------------------------------------------------------------
st.subheader("3. Coverage check")
if policy_row is not None:
    result = check_coverage(
        policy_row,
        loss_date=submission.loss_date,
        loss_type=submission.loss_type.value,
        claim_amount_usd=(
            float(claim_row["total_claim_amount"]) if claim_row is not None else None
        ),
    )
    icon = {
        CoverageStatus.CONFIRMED: "✅",
        CoverageStatus.QUESTION: "⚠️",
        CoverageStatus.NOT_IN_FORCE: "🚫",
        CoverageStatus.UNKNOWN: "❓",
    }[result.status]

    if result.is_confirmed:
        st.success(f"{icon} **{result.status.value}**")
    else:
        st.warning(f"{icon} **{result.status.value}**")

    for reason in result.reasons:
        st.markdown(f"- {reason}")

    if result.blocks_fast_track:
        st.info(
            "This claim cannot be fast-tracked and goes to a human. "
            "**ClaimPilot can auto-approve, but it can never auto-deny** — a "
            "coverage problem is a question for an adjuster, not an automated "
            "rejection.",
            icon="👤",
        )
    simulated_badge("Policy terms are simulated")
else:
    st.info(
        "Pick a sample claim on the right to run the coverage check against its "
        "mock policy.",
        icon="ℹ️",
    )

with st.expander("⚠️ How good is this extractor, really?"):
    st.markdown(
        """
The extractor scores **100% on the 1,000 generated narratives** and on a set of
hand-written paraphrases. **Do not read that as production accuracy.**

The narratives were generated from templates, and the regular expressions were
written against those templates — so the headline figure is close to circular.
The paraphrase set was also used while tuning, which means no genuinely
held-out evaluation exists here: every piece of text in this project is
synthetic.

Treat 100% as *"the regression suite passes"*, not as evidence the extractor
would survive real claimant writing. On real FNOL text — misspellings,
dictation errors, mixed languages, missing punctuation — accuracy would be
materially lower, and a production build would use a language model with a
human-reviewed confidence threshold.
"""
    )

footer()
