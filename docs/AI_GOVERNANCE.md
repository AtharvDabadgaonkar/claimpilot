# AI Governance — ClaimPilot

**Northwind Mutual (fictional carrier) · Personal Auto Claims · Inventory `model-inventory-v0.1`**

> ⚠️ Demo project using public and simulated data. Not a licensed insurance product, not legal or compliance advice.
>
> This programme is **designed around** the NAIC AI Model Bulletin; it is not compliant with, nor certified against, any standard.

---

## 1. Programme summary

ClaimPilot uses AI at four points in the claim lifecycle: extracting structured
data from a policyholder's free-text account, scoring claims for fraud signal,
suggesting reserves, and briefing adjusters. Each component is inventoried with
a documented purpose, data source, inputs, exclusions, limitations and failure
modes (`config/model_cards.yaml`).

Two product rules govern everything else:

1. **The system can auto-approve, but it can never auto-deny.** This is enforced
   structurally, not by policy: the routing `Destination` enum has no denial
   member, the rules file declares a forbidden-destination list, and the engine
   refuses to load a rules file containing one. An exhaustive test routes all
   1,000 claims and asserts no denial outcome exists.
2. **Protected characteristics are never model inputs — and outcomes are tested
   across them anyway.** Exclusion alone does not prevent a disparate outcome.

---

## 2. Risk tiers

| Tier | Meaning | Components |
|---|---|---|
| 🔴 **High** | Can change what happens to a claim | Fraud Signal Model |
| 🟠 **Medium** | Shapes routing or valuation, always under human review | FNOL Extractor, Reserve Model |
| 🟢 **Low** | Advisory only; a human acts on it or ignores it | Adjuster Copilot |

The fraud model is the only high-tier component because a high score moves a
claim into an SIU queue. It still cannot deny, pay, or close anything.

---

## 3. Human-in-the-loop policy

| Outcome | Who decides | Controls |
|---|---|---|
| Fast-track approval (STP) | System | Only when severity, complexity and fraud are all Low **and** coverage is confirmed. 8.1% of claims. |
| Routine handling | Adjuster | AI supplies scores, reasons, reserve range and a document checklist. |
| Complex / senior handling | Senior adjuster | Triggered by high complexity, high severity, or any coverage question. |
| SIU referral | Investigator | A referral is an investigation, never an accusation or a denial. |
| **Denial** | **Human only** | Requires a valid reason code **and** a written note of at least 20 characters. The guard is in the pydantic schema, so no code path can write an unjustified denial. |

A coverage problem — the most likely route to a denial — is the **first** rule
evaluated, and it routes to a senior adjuster rather than closing the claim.

---

## 4. Fairness approach

**What is excluded.** Nine fields never reach any model: `insured_sex`, `age`,
`insured_education_level`, `insured_occupation`, `insured_relationship`,
`insured_zip`, `insured_hobbies`, `policy_number`, `incident_location`. The
fraud model additionally excludes `incident_state`, `incident_city` (geographic
proxies), `auto_make` (wealth proxy) and `capital_gains` / `capital_loss`
(socio-economic proxies). Features are an **allow-list**: a field must be named
in config to reach a model, so a mistake fails closed.

**What is measured.** The STP fast-track rate and the SIU referral rate across
region, age band, vehicle category and sex — the same characteristics the
models may not see. The statistic is the **disparate impact ratio**: lowest
group's rate ÷ highest group's rate, flagged below 0.8. Groups under 30 claims
are reported but excluded from the ratio, because a group of five produces a
rate that would dominate the comparison.

**On the 0.8 threshold.** It is the EEOC four-fifths rule from US *employment*
law, used because it is widely understood. It is not an insurance standard
and not a legal test for claims handling. Insurance discrimination is governed
by state unfair trade practice and unfair claims settlement acts, which define
no single numeric test. A real carrier would agree its metrics with counsel and
its regulators.

**What the test found.** This portfolio **fails 5 of 8 tests**, worst ratio
0.355 (STP rate by age band). Age is not a model input, yet the oldest band is
fast-tracked at roughly a third the rate of the highest band. The path runs
through correlation: that band has the fewest low-severity claims and the
highest reported-fraud rate in this data, so it fails the low-severity and
low-fraud gates more often. The fraud label is itself a *reported* flag, so
historical bias in who was referred is reproduced.

This is the point of outcome testing: **excluding a protected attribute did not
prevent a disparate outcome.** On a real portfolio this result would block
deployment pending investigation, not sit in a demo.

---

## 5. Audit approach

Every AI recommendation and every human decision writes one immutable record
capturing: timestamp and claim ID; the AI's recommendation, bands, fraud
probability and suggested reserve; the versions of the triage rules, routing
rules, fraud model and reserve model that produced it; the inputs used; the
human decision, decider, reason code and note; and **whether the human overrode
the AI**. The trail is filterable in the app and exports to CSV.

**Storage limitation.** The log lives in session state because Streamlit
Community Cloud resets its file system on restart. It is **not permanent**. A
real deployment would write to an append-only store with retention aligned to
state record-keeping requirements.

---

## 6. Ongoing monitoring

| Signal | Thresholds | Why |
|---|---|---|
| Override rate | Alert above 25%, **warn below 2%** | A high rate questions the model. A near-zero rate is not a success — if humans never disagree, the likelier explanation is rubber-stamping, not perfection. |
| Score distributions | Weekly trend | Detects a shift in the claim mix the models see. |
| Drift (PSI) | Warn 0.10, alert 0.25 | Population stability of the fraud score between periods. |

Override rate counts only decisions where the AI made a **substantive** call — a
fast-track approval or an SIU referral. Routing a claim to a queue is the AI
asking for human review; whatever the adjuster then decides, nothing was
overridden.

**Drift caveat.** The signal compares halves of the same two-month period rather
than live scores against the training distribution, over ~1,000 claims. The
mechanism is demonstrated; the number is not dependable.

---

## 7. Privacy and data minimisation

Personal data is masked **before** any text reaches a language model. Masking
happens in the provider base class, so no subclass can bypass it — a test
constructs a subclass and asserts the raw value never reaches it. Masking covers
names, emails, phone numbers, policy numbers, addresses, VINs, plates and SSNs,
measured at 100% detection across 464 narratives known to contain PII with zero
false positives across 536 clean ones.

The copilot reads only a fixed allow-list of 20 claim fields and is structurally
unable to see policyholder demographics. The masker is regex-based and
**demo-grade**; a real deployment would use a validated PII detection service.

No real personal data is used anywhere. Policyholder names are fictional, emails
use the reserved `example.com` domain, and phone numbers use the reserved `555`
area code, so nothing is deliverable.

---

## 8. What a real insurer would need beyond this demo

This project demonstrates the *shape* of an AI governance programme. A carrier
putting something like it into production would need, at minimum:

- **State-by-state rules.** Deadlines, unfair claims settlement practice
  requirements and permitted use of AI vary by jurisdiction. Everything here is
  a single illustrative set.
- **Regulatory filings and notification.** Several states require disclosure of
  AI use in claims handling; some require filings before deployment.
- **Independent model validation.** The evaluation dates here are
  self-assessments. A real programme needs validation by a party independent of
  the builders, with documented challenge.
- **Recalibration on real data.** The fraud rate here is ~25%, roughly 10–25×
  a real auto book. Every threshold would need rebuilding against the carrier's
  own experience, and the reported-fraud label would need replacing with
  adjudicated outcomes.
- **An adverse action and appeals process.** Policyholders need to know a
  decision involved AI, and have a route to human reconsideration.
- **Vendor and third-party due diligence** for any hosted model, including data
  handling terms and the right to audit.
- **Retention, access control and an immutable audit store**, rather than
  session state.
- **A named accountable executive** for the AI programme, with board-level
  reporting on the metrics above.
- **Incident response.** A defined path for withdrawing a model from production
  when monitoring trips.

---

*Every figure in this document is generated by `make metrics` and lives in
`reports/metrics.json`. Nothing here is typed by hand.*
