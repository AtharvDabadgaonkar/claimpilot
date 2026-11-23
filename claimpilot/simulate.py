"""Simulated layers built on top of the real claim rows.

EVERYTHING in this module is SIMULATED and labelled as such in the UI:

* FNOL narratives, written from each row's real field values. The values used
  are retained as ground truth so extraction accuracy can be *measured*
  rather than asserted.
* Mock policies (coverage type, term dates, deductible, limits).
* Fake policyholder contact details, which exist only so the PII-masking
  demo has something to mask. Names are fictional; emails use the reserved
  example.com domain and phones the reserved 555 area code.
* Adjuster handling times, cycle times and costs.
* Claim deadlines (illustrative; real deadlines are state-specific).

All draws come from a single seeded generator, so the output is byte-identical
on every run and the committed Parquet files are reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from claimpilot import config

# -- Fictional people --------------------------------------------------------
# A small fixed list. These are invented names; any resemblance to a real
# person is coincidental and unintended.
_FIRST_NAMES = [
    "Jordan", "Riley", "Avery", "Casey", "Morgan", "Quinn", "Reese", "Rowan",
    "Sawyer", "Harper", "Emerson", "Finley", "Hayden", "Kendall", "Logan",
    "Parker", "Peyton", "Skyler", "Tatum", "Blake", "Dana", "Elliot",
    "Marlowe", "Noor", "Priya", "Rafael", "Sofia", "Theo", "Vera", "Omar",
]
_LAST_NAMES = [
    "Alvarez", "Bennett", "Chen", "Delgado", "Ellis", "Faulkner", "Gupta",
    "Hollis", "Ibrahim", "Jensen", "Kowalski", "Lindqvist", "Mbeki",
    "Novak", "Okafor", "Pereira", "Quigley", "Rossi", "Sandoval", "Tran",
    "Ueda", "Vargas", "Whitfield", "Xu", "Yamamoto", "Zielinski",
]

# -- Narrative phrasing ------------------------------------------------------
_OPENERS = [
    "I was {driving_ctx} when {event}.",
    "{event} while I was {driving_ctx}.",
    "This is to report that {event}.",
    "Filing a claim because {event}.",
]

_EVENT_BY_TYPE = {
    "Single Vehicle Collision": [
        "I lost control and hit {obstacle}",
        "my car went off the road and struck {obstacle}",
        "I skidded and collided with {obstacle}",
    ],
    "Multi-vehicle Collision": [
        "{n} vehicles were involved in a pile-up",
        "I was hit by another car and then pushed into a {nth} vehicle",
        "a chain-reaction crash involving {n} cars occurred",
    ],
    "Vehicle Theft": [
        "my vehicle was stolen from where I had parked it",
        "I came back and my car was gone",
        "someone took my vehicle overnight",
    ],
    "Parked Car": [
        "someone hit my parked car and left",
        "my car was damaged while parked and unattended",
        "I found damage to my parked vehicle",
    ],
}

_NO_COLLISION_TYPES = {"Vehicle Theft", "Parked Car"}

# The ordinal must match the real vehicle count, otherwise the count is
# not recoverable from the narrative and ground truth is unachievable.
_ORDINALS = {2: "second", 3: "third", 4: "fourth", 5: "fifth"}

_OBSTACLES = ["a guardrail", "a utility pole", "a concrete barrier", "a tree", "a road sign"]
_DRIVING_CTX = [
    "on my way home from work", "driving to the store", "heading to work",
    "on the highway", "driving through town", "on my usual commute",
]
_COLLISION_PHRASE = {
    "Rear Collision": "the impact was to the rear of my vehicle",
    "Side Collision": "I was struck on the side",
    "Front Collision": "the front end took the impact",
}
_SEVERITY_PHRASE = {
    "Trivial Damage": "The damage looks very minor, mostly cosmetic.",
    "Minor Damage": "There is visible damage but the car is still drivable.",
    "Major Damage": "The damage is extensive and the car is not drivable.",
    "Total Loss": "The vehicle appears to be a total loss.",
}
_VAGUE_AMOUNTS = [
    "a few hundred dollars", "maybe a couple thousand dollars",
    "several thousand dollars", "quite a lot, I am not sure how much",
    "I have no idea what it will cost",
]


def _rng() -> np.random.Generator:
    return np.random.default_rng(config.random_seed())


# ---------------------------------------------------------------------------
# Contact details (fictional)
# ---------------------------------------------------------------------------
def simulate_contacts(claims: pd.DataFrame) -> pd.DataFrame:
    """Invent a fictional policyholder contact record per claim."""
    rng = _rng()
    n = len(claims)
    domain = config.assumption("mock_contact", "email_domain")
    area = int(config.assumption("mock_contact", "phone_area_codes")[0])

    first = rng.choice(_FIRST_NAMES, size=n)
    last = rng.choice(_LAST_NAMES, size=n)
    full = [f"{fn} {ln}" for fn, ln in zip(first, last, strict=False)]

    return pd.DataFrame(
        {
            "claim_id": claims["claim_id"].to_numpy(),
            "policyholder_name": full,
            "policyholder_email": [
                f"{fn.lower()}.{ln.lower()}@{domain}"
                for fn, ln in zip(first, last, strict=False)
            ],
            "policyholder_phone": [
                f"({area}) {rng.integers(200, 1000)}-{rng.integers(1000, 10000)}"
                for _ in range(n)
            ],
            "is_simulated": True,
        }
    )


# ---------------------------------------------------------------------------
# Mock policies
# ---------------------------------------------------------------------------
def simulate_policies(claims: pd.DataFrame) -> pd.DataFrame:
    """Build a mock policy per claim, including some genuine coverage problems.

    A deliberate minority of policies are expired at the date of loss, or are
    liability-only against a physical-damage claim, so the coverage check has
    real failures to catch rather than always passing.
    """
    rng = _rng()
    cfg = config.assumption("mock_policy")
    n = len(claims)

    term_months = int(cfg["term_months"])
    expired_rate = float(cfg["expired_policy_rate"])
    liability_only_rate = float(cfg["liability_only_rate"])
    prefix = cfg["policy_number_prefix"]

    bind = pd.to_datetime(claims["policy_bind_date"])

    # Each policy renews annually from its bind date; take the term that should
    # contain the loss, then force a minority to be expired.
    loss = pd.to_datetime(claims["incident_date"])
    term_start = pd.to_datetime(
        {
            "year": loss.dt.year,
            "month": bind.dt.month,
            "day": bind.dt.day.clip(upper=28),
        }
    )
    # If that start is after the loss, the live term began the previous year.
    term_start = term_start.where(
        term_start <= loss, term_start - pd.DateOffset(years=1)
    )
    term_end = term_start + pd.DateOffset(months=term_months)

    is_expired = rng.random(n) < expired_rate
    # For the expired cases, end the term before the loss date.
    lapse_gap = rng.integers(5, 120, size=n)
    term_end_final = term_end.copy()
    term_end_final[is_expired] = (
        loss[is_expired] - pd.to_timedelta(lapse_gap[is_expired], unit="D")
    )
    term_start_final = term_end_final - pd.DateOffset(months=term_months)

    liability_only = rng.random(n) < liability_only_rate
    coverage = np.where(
        liability_only, "Liability Only", "Comprehensive + Collision + Liability"
    )

    return pd.DataFrame(
        {
            "claim_id": claims["claim_id"].to_numpy(),
            "mock_policy_number": [
                f"{prefix}-{int(p)}" for p in claims["policy_number"].to_numpy()
            ],
            "coverage_type": coverage,
            "term_start": term_start_final.to_numpy(),
            "term_end": term_end_final.to_numpy(),
            "deductible_usd": claims["policy_deductable"].to_numpy(),
            "csl_per_person_usd": claims["csl_per_person_usd"].to_numpy(),
            "csl_per_occurrence_usd": claims["csl_per_occurrence_usd"].to_numpy(),
            "annual_premium_usd": claims["policy_annual_premium"].to_numpy(),
            "umbrella_limit_usd": claims["umbrella_limit"].to_numpy(),
            # Convenience flags the coverage check re-derives rather than trusts.
            "expected_expired_at_loss": is_expired,
            "expected_liability_only": liability_only,
            "is_simulated": True,
        }
    )


# ---------------------------------------------------------------------------
# FNOL narratives, with ground truth retained
# ---------------------------------------------------------------------------
def _build_narrative(
    row: pd.Series,
    contact: pd.Series,
    rng: np.random.Generator,
    include_pii: bool,
    omit_fields: bool,
    vague_amount: bool,
) -> tuple[str, dict]:
    """Write one FNOL description and return it with its ground-truth fields."""
    itype = row["incident_type"]
    sev = row["incident_severity"]
    ctype = row["collision_type"]
    n_veh = int(row["number_of_vehicles_involved"])

    event_tpl = rng.choice(_EVENT_BY_TYPE.get(itype, _EVENT_BY_TYPE["Parked Car"]))
    event = event_tpl.format(
        obstacle=rng.choice(_OBSTACLES),
        n=n_veh if n_veh > 1 else 2,
        nth=_ORDINALS.get(n_veh, "another"),
    )
    opener = rng.choice(_OPENERS).format(
        driving_ctx=rng.choice(_DRIVING_CTX), event=event
    )
    opener = opener[0].upper() + opener[1:]

    sentences = [opener]

    # Date and place of loss.
    loss_date = pd.to_datetime(row["incident_date"])
    sentences.append(
        f"It happened on {loss_date.strftime('%B %d, %Y')} at around "
        f"{int(row['incident_hour_of_the_day'])}:00 in "
        f"{row['incident_city']}, {row['incident_state']}."
    )

    # Collision direction, where applicable.
    if ctype in _COLLISION_PHRASE and not omit_fields:
        phrase = _COLLISION_PHRASE[ctype]
        sentences.append(phrase[0].upper() + phrase[1:] + ".")

    # Severity.
    sentences.append(_SEVERITY_PHRASE.get(sev, "There is damage to the vehicle."))

    # Injuries.
    injuries = int(row["bodily_injuries"])
    if injuries > 0:
        sentences.append(
            f"{injuries} {'person was' if injuries == 1 else 'people were'} "
            "injured and needed medical attention."
        )
    elif not omit_fields:
        sentences.append("Nobody was injured.")

    # Police report.
    police_known = pd.notna(row["police_report_available"])
    if police_known and not omit_fields:
        if bool(row["police_report_available"]):
            sentences.append("The police attended and a report was filed.")
        else:
            sentences.append("No police report was taken.")

    # Witnesses.
    witnesses = int(row["witnesses"])
    if witnesses > 0 and not omit_fields:
        sentences.append(
            f"There {'was' if witnesses == 1 else 'were'} {witnesses} "
            f"{'witness' if witnesses == 1 else 'witnesses'}."
        )

    # Vehicle.
    sentences.append(
        f"The vehicle is a {int(row['auto_year'])} {row['auto_make']} {row['auto_model']}."
    )

    # Amount: either the real figure or a vague phrase.
    amount = float(row["total_claim_amount"])
    if vague_amount:
        sentences.append(f"I think the repairs will cost {rng.choice(_VAGUE_AMOUNTS)}.")
        stated_amount = None
    else:
        sentences.append(f"I estimate the total damage at about ${amount:,.0f}.")
        stated_amount = amount

    # Contact details — present in a minority of narratives, to exercise masking.
    if include_pii:
        sentences.append(
            f"My name is {contact['policyholder_name']} and my policy number is "
            f"{contact['mock_policy_number']}. You can reach me at "
            f"{contact['policyholder_email']} or {contact['policyholder_phone']}. "
            f"I live at {row['incident_location']}."
        )

    text = " ".join(sentences)

    # Ground truth: what a perfect extractor should return FROM THIS TEXT.
    #
    # This is deliberately not "the underlying row value". If the narrative
    # never mentions witnesses, the correct extraction is None ("not stated"),
    # not 0 — silence is a documentation gap, not a negative answer. Scoring
    # against the row value instead would penalise the extractor for refusing
    # to invent facts, which is exactly the behaviour we want.
    collision_stated = ctype in _COLLISION_PHRASE and not omit_fields
    injury_stated = injuries > 0 or not omit_fields
    witnesses_stated = witnesses > 0 and not omit_fields

    truth = {
        "loss_date": loss_date.date().isoformat(),
        "loss_type": itype,
        # A theft or parked-car claim has no collision direction to state, and
        # that is derivable from the text itself.
        "collision_type": (
            ctype
            if collision_stated
            else ("Not applicable" if itype in _NO_COLLISION_TYPES else None)
        ),
        "severity_stated": sev,
        "injury_reported": (injuries > 0) if injury_stated else None,
        "num_injured": injuries if injury_stated else None,
        "police_report_available": (
            bool(row["police_report_available"]) if police_known and not omit_fields else None
        ),
        "num_vehicles": n_veh,
        "witnesses": witnesses if witnesses_stated else None,
        "city": row["incident_city"],
        "state": row["incident_state"],
        "auto_make": row["auto_make"],
        "auto_year": int(row["auto_year"]),
        "stated_amount_usd": stated_amount,
        "contains_pii": include_pii,
    }
    return text, truth


def simulate_narratives(
    claims: pd.DataFrame, policies: pd.DataFrame, contacts: pd.DataFrame
) -> pd.DataFrame:
    """Generate one FNOL narrative per claim, with ground-truth fields."""
    rng = _rng()
    cfg = config.assumption("narratives")
    n = len(claims)

    include_pii = rng.random(n) < float(cfg["include_pii_rate"])
    omit_fields = rng.random(n) < float(cfg["omit_field_rate"])
    vague = rng.random(n) < float(cfg["vague_amount_rate"])

    merged = contacts.merge(
        policies[["claim_id", "mock_policy_number"]], on="claim_id", how="left"
    )

    rows = []
    for i in range(n):
        text, truth = _build_narrative(
            claims.iloc[i],
            merged.iloc[i],
            rng,
            bool(include_pii[i]),
            bool(omit_fields[i]),
            bool(vague[i]),
        )
        rows.append(
            {
                "claim_id": claims.iloc[i]["claim_id"],
                "narrative": text,
                "char_count": len(text),
                "is_simulated": True,
                **{f"truth_{k}": v for k, v in truth.items()},
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Deadlines
# ---------------------------------------------------------------------------
def simulate_deadlines(claims: pd.DataFrame) -> pd.DataFrame:
    """Attach illustrative acknowledge / decide / pay deadlines to each claim.

    Real deadlines are set by state unfair-claims-practices rules and vary by
    jurisdiction and claim type. These are placeholders for the demo only.
    """
    cfg = config.assumption("deadlines")
    reported = pd.to_datetime(claims["incident_date"])

    return pd.DataFrame(
        {
            "claim_id": claims["claim_id"].to_numpy(),
            "reported_date": reported.to_numpy(),
            "acknowledge_due": (
                reported + pd.to_timedelta(int(cfg["acknowledge_days"]), unit="D")
            ).to_numpy(),
            "decide_due": (
                reported + pd.to_timedelta(int(cfg["decide_days"]), unit="D")
            ).to_numpy(),
            "pay_due": (
                reported
                + pd.to_timedelta(
                    int(cfg["decide_days"]) + int(cfg["pay_days"]), unit="D"
                )
            ).to_numpy(),
            "deadline_basis": cfg["label"],
            "is_simulated": True,
        }
    )


# ---------------------------------------------------------------------------
# Handling time and cost (both arms of the backtest)
# ---------------------------------------------------------------------------
def _lognormal_around(
    rng: np.random.Generator, mean: np.ndarray, sigma: float
) -> np.ndarray:
    """Draw positive values with the given mean and lognormal dispersion."""
    mean = np.asarray(mean, dtype=float)
    mu = np.log(np.maximum(mean, 1e-6)) - (sigma**2) / 2
    return rng.lognormal(mu, sigma)


def simulate_manual_baseline(claims: pd.DataFrame) -> pd.DataFrame:
    """The 'everything manual' comparison arm: no triage, no automation.

    Only the baseline is generated in Phase 1. The ClaimPilot arm depends on
    triage and routing decisions, so it is built in Phase 5's backtest.
    """
    rng = _rng()
    ht = config.assumption("handling_time_hours", "manual_baseline")
    ct = config.assumption("cycle_time_days", "manual_baseline")
    costs = config.assumption("costs")

    sev = claims["incident_severity"]
    band = pd.Series("medium_severity", index=claims.index)
    band[sev.isin(["Trivial Damage", "Minor Damage"])] = "low_severity"
    band[sev.isin(["Major Damage", "Total Loss"])] = "high_severity"

    hours_mean = band.map(
        {k: float(ht[k]) for k in ("low_severity", "medium_severity", "high_severity")}
    ).to_numpy()
    days_mean = band.map(
        {k: float(ct[k]) for k in ("low_severity", "medium_severity", "high_severity")}
    ).to_numpy()

    hours = _lognormal_around(rng, hours_mean, float(ht["sigma"]))
    days = _lognormal_around(rng, days_mean, float(ct["sigma"]))

    lae = (
        hours * float(costs["adjuster_cost_per_hour_usd"])
        + float(costs["fixed_cost_per_claim_usd"])
    )

    return pd.DataFrame(
        {
            "claim_id": claims["claim_id"].to_numpy(),
            "severity_band": band.to_numpy(),
            "baseline_handling_hours": np.round(hours, 2),
            "baseline_cycle_days": np.round(days, 1),
            "baseline_lae_usd": np.round(lae, 2),
            "baseline_time_to_first_contact_hours": float(
                config.assumption(
                    "cycle_time_days", "time_to_first_contact_hours", "manual_baseline"
                )
            ),
            "is_simulated": True,
        }
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run() -> dict[str, pd.DataFrame]:
    """Build every simulated table and write the committed Parquet files."""
    from claimpilot.ingest import load_claims

    claims = load_claims()

    contacts = simulate_contacts(claims)
    policies = simulate_policies(claims)
    narratives = simulate_narratives(claims, policies, contacts)
    deadlines = simulate_deadlines(claims)
    baseline = simulate_manual_baseline(claims)

    # Contacts are folded into the policies table; they are only ever used for
    # the masking demo and are never model inputs. Drop the duplicated
    # is_simulated flag so the merge does not produce _x/_y suffixes.
    policies = policies.merge(
        contacts.drop(columns=["is_simulated"]), on="claim_id", how="left"
    )

    config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    policies.to_parquet(config.POLICIES_PARQUET, index=False)
    narratives.to_parquet(config.NARRATIVES_PARQUET, index=False)
    deadlines.to_parquet(config.SAMPLE_DIR / "deadlines.parquet", index=False)
    baseline.to_parquet(config.SAMPLE_DIR / "baseline.parquet", index=False)

    for name, df in (
        ("policies", policies),
        ("narratives", narratives),
        ("deadlines", deadlines),
        ("baseline", baseline),
    ):
        path = config.SAMPLE_DIR / f"{name}.parquet"
        print(f"  {name:<11} {len(df):>5} rows  {path.stat().st_size / 1024:>6.0f} KB")

    return {
        "claims": claims,
        "policies": policies,
        "narratives": narratives,
        "deadlines": deadlines,
        "baseline": baseline,
    }


def load_narratives() -> pd.DataFrame:
    return pd.read_parquet(config.NARRATIVES_PARQUET)


def load_policies() -> pd.DataFrame:
    return pd.read_parquet(config.POLICIES_PARQUET)
