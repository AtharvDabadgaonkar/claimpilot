# ClaimPilot — Product Requirements Document

**Status:** demo / portfolio build, v0.1
**Carrier:** Northwind Mutual (fictional)
**Line of business:** US personal auto, first-party and third-party
**Disclaimer:** Demo project using public and simulated data. Not a licensed insurance product, not legal or compliance advice.

---

## 1. Problem

A personal auto claims operation is pulled three ways at once:

1. **Cycle time.** Policyholders judge a carrier almost entirely on how fast a claim closes. Every day of delay costs retention and invites a complaint.
2. **Claim leakage.** Money paid that shouldn't have been — inflated estimates, uninvestigated red flags, coverage never properly checked. Industry estimates put leakage in the low double digits of indemnity spend.
3. **Fairness and conduct.** Unfair claims settlement practice acts, and now the NAIC AI Model Bulletin, make the *process* a regulated object, not just the outcome.

Pushing on any one of these degrades the other two. Automate aggressively and leakage and conduct risk rise. Review everything manually and cycle time and expense rise. The carriers that win are the ones that can say precisely *which* claims they automated, *why*, and *what they gave up*.

Most claims-AI demos show only the first axis. ClaimPilot is built to show the trade-off.

## 2. Users

| User | What they need | Where it lands in the product |
|---|---|---|
| **Policyholder** | File a claim in minutes, know it was received, get a decision fast | Digital FNOL intake with immediate acknowledgement |
| **Claims adjuster** | Know which file to pick up next, and not retype what the file already says | Workbench queue by priority, copilot briefing, reserve suggestion |
| **Claims manager** | Throughput and cycle time without a conduct incident | Claims Ops Dashboard — KPIs with guardrails beside them |
| **SIU investigator** | A referral queue with the reasons attached, not a score | SIU Queue with named red flags per claim |
| **Compliance officer** | Evidence of oversight: what the AI recommended, what a human decided, and whether outcomes differ by group | Governance & Audit — audit trail, fairness tests, model inventory, monitoring |

## 3. Goals

**G1. Triage every claim on three independent axes** — severity, complexity, fraud signal — each with human-readable reason codes, never a single opaque score.

**G2. Fast-track only the claims that clear every gate.** Low severity, low complexity, low fraud signal, coverage confirmed. All four, or a person looks at it.

**G3. Never auto-deny.** The system may auto-approve. Every potential denial reaches a human. This is enforced structurally, not by policy: the routing destination enum has no denial member and a test asserts the config cannot introduce one.

**G4. Assist the adjuster without inventing anything.** The copilot sees an explicit 20-field allow-list from the claim's own record. Reserve figures come from a quantile model, never from a language model.

**G5. Make the AI auditable.** Every decision records the AI recommendation, the human decision, a reason code, and whether the human overrode the model. Adverse decisions require a written note.

**G6. Test outcomes, not just inputs.** Excluding protected attributes is necessary and not sufficient. Disparate impact is measured on the fast-track rate across groups, and reported whether or not it passes.

**G7. Publish no unreproducible number.** `make metrics` writes every figure to `reports/metrics.json`; the README and the app read from it. A test asserts the README's numbers match the file.

## 4. Non-goals

- **Not a rating or underwriting engine.** Pricing, eligibility and renewal are out of scope.
- **Not a payments system.** No disbursement, no vendor payment, no recovery ledger.
- **Not a document or photo pipeline.** No OCR, no image damage estimation.
- **Not a claims system of record.** No policy admin integration, no persistent storage; the deployed demo holds new claims and decisions in session state only.
- **Not a compliance product.** The governance module is *designed around* the NAIC AI Model Bulletin. It is not an attestation and confers no compliance status.
- **Not a measured efficiency claim.** The backtest is assumption-driven. See §7.

## 5. MVP scope (built)

| Module | Scope |
|---|---|
| **1. Digital FNOL intake** | Validated submission (pydantic), plus a rule-based extractor that reads a free-text narrative into structured fields and reports what the text does *not* state |
| **2. Triage & routing** | Severity, complexity and fraud scores with reason codes; a gradient-boosted fraud model on an allow-listed feature set; coverage check against mock policy terms; YAML-configured routing with four destinations and no denial path |
| **3. Adjuster workbench** | Priority queue (deadline, severity, fraud, complexity, age), copilot briefing from an allow-listed field set, quantile reserve suggestion with a P10–P90 band, decision form with reason codes and mandatory denial notes |
| **4. Governance & compliance** | Audit trail with override detection, disparate-impact testing across four group dimensions, a four-component model inventory with risk tiers, override-rate and population-stability monitoring |
| **5. Claims ops dashboard** | KPI tiles with guardrails beside them, and a paired backtest against an all-manual arm with a break-even analysis |

Cross-cutting: PII masking is enforced at the LLM boundary in the abstract base class, so no provider implementation can bypass it. The LLM is optional — with no API key the app uses a rule-based mock and shows a "Demo mode: rule-based AI" badge.

## 6. KPIs

**Efficiency**
- Straight-through processing rate
- Mean cycle time, FNOL to close
- Time to first contact
- LAE per claim
- Adjuster FTE-months for the book

**Quality**
- Reserve accuracy: MAE and P10–P90 interval coverage
- SIU referral hit rate against the reported-fraud flag
- FNOL extraction field accuracy

**Guardrails — reported beside every efficiency number, never after it**
- Estimated claim leakage
- Reopen rate
- Worst disparate-impact ratio and the count of flagged tests
- Override rate, with both a ceiling (model distrusted) and a floor (rubber-stamping)
- Deadline breach rate
- Claims auto-denied — which must be zero

## 7. Risks and what the demo does about them

| Risk | Why it matters | Mitigation in this build |
|---|---|---|
| **The dataset is not a portfolio.** ~1,000 claims, 24.7% reported-fraud rate, median claim ~$58k | Every threshold tuned here is wrong for a real book, and the referral volume is an order of magnitude too high | Stated on every page and in the README; the dashboard's break-even analysis quantifies exactly how the fraud rate drives the result |
| **The backtest is assumption-driven** | Arms are paired on the same random draw, so deltas are arithmetic consequences of `config/assumptions.yaml`, not evidence | Labelled as such on the page itself, in the module docstring, and in `metrics.json`; the only measured figures are the routing mix and the SIU hit rate |
| **The fraud label is a *reported* flag** | Training on it reproduces historical referral bias, and "hit rate" measures agreement with that history rather than accuracy | Named in the model card, the SIU tab and the fairness analysis; the fairness failure is traced to this label |
| **Excluding protected attributes is not enough** | Proxies carry the signal; the build demonstrates this rather than asserting it | Outcome testing across four group dimensions; 5 of 8 tests currently flag and the threshold was left alone |
| **Extraction accuracy is near-circular** | Narratives are template-generated and the regexes were written against those templates | The caveat is in `metrics.json` and the model card; the figure is presented as a regression suite, not an accuracy estimate |
| **The 0.80 fairness threshold is borrowed** | It is the EEOC four-fifths rule from employment law, not an insurance standard | Qualified in the config, the app and `AI_GOVERNANCE.md`; insurance conduct is governed by state acts that define no single numeric test |
| **Demo storage is not permanent** | Streamlit Community Cloud resets its file system on restart | New claims and decisions live in session state, the UI says so, and the audit log exports to CSV |

## 8. Phase 2 roadmap

**Coverage and lines of business**
- Property and homeowners claims — a different severity model, peril-specific coverage logic, and catastrophe surge handling
- Commercial auto and fleet

**Automation depth**
- Photo damage estimation from first-notice images, with an explicit human review band
- Document intake: police reports, repair estimates, medical records via OCR plus extraction
- Subrogation identification and recovery tracking — currently absent, and a material leakage source

**Integration**
- Payments: disbursement to claimant and vendor, with reconciliation
- Guidewire-style ClaimCenter integration — claim sync, activity creation, and writing triage output back onto the file as a note rather than a hidden score
- Policy admin for real coverage verification, replacing the mock policy layer

**Governance maturity**
- Recalibration against the carrier's own experience, replacing every threshold in this build
- Independent model validation and a named accountable executive per the NAIC bulletin
- State-by-state deadline rules, replacing the illustrative single set
- Adverse-action notices and an appeals path for claimants
- Regulatory filings and state-specific AI governance disclosures where required
- Immutable audit storage with retention, replacing session state

**Measurement**
- A real backtest: replay decisions against observed closed-claim outcomes, with confidence intervals
- A/B holdback — a share of claims handled without triage, so the saving is measured rather than modelled
