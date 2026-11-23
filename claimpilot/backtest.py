"""Backtest: the ClaimPilot arm against an 'everything manual' arm.

Read this before reading any number the backtest produces.

The two arms are **paired**. The manual arm's handling hours and cycle days
were drawn once, in Phase 1, from the lognormal distributions in
``config/assumptions.yaml``. The ClaimPilot arm is built from *those same
draws* by applying the multipliers for the route each claim actually took. No
second random draw happens anywhere in this module.

That design is deliberate, and it has a consequence worth stating plainly:

    Every delta this module reports is an arithmetic consequence of the
    assumptions in config/assumptions.yaml. It is not evidence of a saving.

Pairing removes sampling noise, so the deltas are stable and reproducible --
but stability is not validity. A real backtest replays decisions against
*observed* outcomes. This one replays decisions against *assumed* outcomes,
because no real handling times, costs or leakage exist in the public dataset.
The honest claim is: "here is the operating model, here are its assumptions,
and here is exactly how the headline numbers fall out of them."

What *is* measured from the data, not assumed:

* which claims the routing engine actually sent where;
* the SIU referral hit rate (precision against the reported-fraud flag);
* the share of flagged claims the engine failed to refer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from claimpilot import config

BACKTEST_PARQUET = config.SAMPLE_DIR / "backtest.parquet"
BACKTEST_VERSION = "backtest-paired-v0.1"

STP = "stp_auto_approve"
SIU = "siu_queue"


# ---------------------------------------------------------------------------
# The ClaimPilot arm
# ---------------------------------------------------------------------------
def build_arms(
    baseline: pd.DataFrame, scored: pd.DataFrame, claims: pd.DataFrame
) -> pd.DataFrame:
    """Derive the ClaimPilot arm from each claim's own baseline draw.

    Returns one row per claim with both arms side by side.
    """
    ht = config.assumption("handling_time_hours", "claimpilot")
    ct = config.assumption("cycle_time_days", "claimpilot")
    costs = config.assumption("costs")
    tfc = config.assumption("cycle_time_days", "time_to_first_contact_hours")

    df = (
        baseline.merge(
            scored[["claim_id", "destination", "auto_approved", "fraud_band"]],
            on="claim_id",
            how="inner",
        )
        .merge(
            claims[["claim_id", "total_claim_amount", "fraud_reported"]],
            on="claim_id",
            how="inner",
        )
        .reset_index(drop=True)
    )
    if len(df) != len(baseline):
        raise ValueError(
            f"Backtest arms lost rows: {len(baseline)} baseline claims but "
            f"{len(df)} after joining scores and claims."
        )

    is_stp = df["destination"].eq(STP).to_numpy()
    is_siu = df["destination"].eq(SIU).to_numpy()

    base_hours = df["baseline_handling_hours"].to_numpy(dtype=float)
    base_days = df["baseline_cycle_days"].to_numpy(dtype=float)

    # --- adjuster touch time -------------------------------------------------
    # Fast-tracked claims collapse to a near-zero human touch. Everything else
    # keeps its own baseline draw, scaled by the assisted-handling multiplier.
    adjuster_hours = np.where(
        is_stp,
        float(ht["stp_auto_approved"]),
        base_hours * float(ht["adjuster_assisted_multiplier"]),
    )
    # SIU time is additional, and costs more per hour than adjuster time.
    siu_hours = np.where(is_siu, float(ht["siu_referral_extra"]), 0.0)

    # --- cycle time ----------------------------------------------------------
    cycle_days = np.where(
        is_stp,
        float(ct["stp_auto_approved"]),
        base_days * float(ct["adjuster_assisted_multiplier"]),
    )
    cycle_days = cycle_days + np.where(is_siu, float(ct["siu_referral_extra"]), 0.0)

    # --- loss adjustment expense --------------------------------------------
    # A fast-tracked claim pays the automated system cost instead of the manual
    # intake/correspondence fixed cost (assumptions: backtest.stp_replaces_fixed_cost).
    fixed = np.where(
        is_stp,
        float(costs["stp_system_cost_per_claim_usd"]),
        float(costs["fixed_cost_per_claim_usd"]),
    )
    lae = (
        adjuster_hours * float(costs["adjuster_cost_per_hour_usd"])
        + siu_hours * float(costs["siu_investigator_cost_per_hour_usd"])
        + fixed
    )

    df["claimpilot_adjuster_hours"] = np.round(adjuster_hours, 2)
    df["claimpilot_siu_hours"] = np.round(siu_hours, 2)
    df["claimpilot_handling_hours"] = np.round(adjuster_hours + siu_hours, 2)
    df["claimpilot_cycle_days"] = np.round(cycle_days, 1)
    df["claimpilot_lae_usd"] = np.round(lae, 2)
    df["claimpilot_time_to_first_contact_hours"] = float(tfc["claimpilot"])
    df["routed_to_siu"] = is_siu
    df["fast_tracked"] = is_stp

    # --- leakage -------------------------------------------------------------
    df = _add_leakage(df)
    df["is_simulated"] = True
    df["backtest_version"] = BACKTEST_VERSION
    return df


def _add_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Estimated avoidable overpayment per claim, per arm.

    Crude by construction. Leakage is modelled as:

        recoverable value  ×  probability the claim was missed

    where recoverable value is a fixed share of the paid amount. The manual
    arm's miss probability is a flat assumed rate. The ClaimPilot arm's is
    derived from what routing actually did with the claim.
    """
    lk = config.assumption("leakage")
    recoverable = float(lk["recoverable_share_of_claim"])
    detection = float(lk["siu_detection_rate"])

    flagged = df["fraud_reported"].astype(bool).to_numpy()
    amount = df["total_claim_amount"].to_numpy(dtype=float)
    exposure = amount * recoverable

    # Manual arm: no triage, so one flat assumed miss rate for flagged claims.
    manual_miss = np.where(flagged, float(lk["manual_missed_fraud_rate"]), 0.0)

    # ClaimPilot arm: a flagged claim not referred is missed outright; a
    # referred one is missed only when detection fails.
    referred = df["routed_to_siu"].to_numpy()
    cp_miss = np.where(flagged, np.where(referred, 1.0 - detection, 1.0), 0.0)

    # Fast-tracking without review carries its own cost, on every claim it
    # touches, flagged or not.
    stp_rate = float(lk["stp_overpayment_rate"])
    stp_leak = np.where(df["fast_tracked"].to_numpy(), amount * stp_rate, 0.0)

    df["baseline_leakage_usd"] = np.round(exposure * manual_miss, 2)
    df["claimpilot_leakage_usd"] = np.round(exposure * cp_miss + stp_leak, 2)
    df["fraud_missed_baseline"] = flagged & (manual_miss > 0)
    df["fraud_referred_claimpilot"] = flagged & referred
    return df


# ---------------------------------------------------------------------------
# Portfolio summary
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmTotals:
    """One arm's portfolio-level outcome."""

    arm: str
    claims: int
    mean_handling_hours: float
    mean_cycle_days: float
    mean_lae_usd: float
    total_lae_usd: float
    time_to_first_contact_hours: float
    total_leakage_usd: float
    mean_leakage_usd: float
    total_cost_usd: float
    reopen_rate: float
    adjuster_fte_months: float
    claims_per_fte_month: float


def _totals(df: pd.DataFrame, arm: str) -> ArmTotals:
    prefix = "baseline" if arm == "manual" else "claimpilot"
    hours = df[f"{prefix}_handling_hours"]
    lae = df[f"{prefix}_lae_usd"]
    leak = df[f"{prefix}_leakage_usd"]
    reopen_key = "reopen_rate_manual" if arm == "manual" else "reopen_rate_claimpilot"
    fte_hours = float(config.assumption("backtest", "adjuster_hours_per_month"))
    fte_months = float(hours.sum()) / fte_hours

    return ArmTotals(
        arm=arm,
        claims=int(len(df)),
        mean_handling_hours=round(float(hours.mean()), 2),
        mean_cycle_days=round(float(df[f"{prefix}_cycle_days"].mean()), 1),
        mean_lae_usd=round(float(lae.mean()), 2),
        total_lae_usd=round(float(lae.sum()), 2),
        time_to_first_contact_hours=round(
            float(df[f"{prefix}_time_to_first_contact_hours"].mean()), 1
        ),
        total_leakage_usd=round(float(leak.sum()), 2),
        mean_leakage_usd=round(float(leak.mean()), 2),
        total_cost_usd=round(float(lae.sum() + leak.sum()), 2),
        reopen_rate=float(config.assumption("leakage", reopen_key)),
        adjuster_fte_months=round(fte_months, 1),
        claims_per_fte_month=round(len(df) / fte_months, 1) if fte_months else 0.0,
    )


def arm_totals(df: pd.DataFrame) -> dict[str, ArmTotals]:
    return {"manual": _totals(df, "manual"), "claimpilot": _totals(df, "claimpilot")}


def comparison_frame(df: pd.DataFrame) -> pd.DataFrame:
    """A tidy two-row frame for the dashboard table."""
    totals = arm_totals(df)
    rows = []
    for key, label in (("manual", "All manual"), ("claimpilot", "ClaimPilot")):
        t = totals[key]
        rows.append(
            {
                "Arm": label,
                "Mean touch time (h)": t.mean_handling_hours,
                "Mean cycle time (days)": t.mean_cycle_days,
                "Time to first contact (h)": t.time_to_first_contact_hours,
                "Mean LAE ($)": t.mean_lae_usd,
                "Est. leakage, total ($)": t.total_leakage_usd,
                "LAE + leakage, total ($)": t.total_cost_usd,
                "Adjuster FTE-months": t.adjuster_fte_months,
                "Reopen rate": t.reopen_rate,
            }
        )
    return pd.DataFrame(rows)


def siu_hit_rate(df: pd.DataFrame) -> dict[str, Any]:
    """Precision and recall of SIU referral against the reported-fraud flag.

    MEASURED, not assumed. The caveat is that ``fraud_reported`` is a
    *reported* flag, not an adjudicated finding, so this is agreement with a
    historical label rather than accuracy against truth.
    """
    referred = df["routed_to_siu"].astype(bool)
    flagged = df["fraud_reported"].astype(bool)
    n_ref = int(referred.sum())
    n_flag = int(flagged.sum())
    hits = int((referred & flagged).sum())
    return {
        "referred": n_ref,
        "referral_rate": round(n_ref / len(df), 4) if len(df) else 0.0,
        "flagged_in_portfolio": n_flag,
        "hits": hits,
        "hit_rate_precision": round(hits / n_ref, 4) if n_ref else 0.0,
        "flagged_caught_recall": round(hits / n_flag, 4) if n_flag else 0.0,
        "portfolio_flag_rate": round(n_flag / len(df), 4) if len(df) else 0.0,
        "flagged_not_referred": n_flag - hits,
        "label_caveat": (
            "fraud_reported is a reported flag, not an adjudicated finding. "
            "Precision above the portfolio flag rate means referrals are "
            "better than random; it does not mean the referrals were right."
        ),
    }


def by_route(df: pd.DataFrame) -> pd.DataFrame:
    """Per-route means for both arms — this is where the headline comes from.

    The portfolio average hides the mechanism: fast-tracked claims collapse to
    almost nothing, while SIU referrals cost *more* than the manual arm would
    have spent, because the manual arm never investigates at all.
    """
    g = df.groupby("destination", dropna=False)
    out = pd.DataFrame(
        {
            "Claims": g.size(),
            "Share": g.size() / len(df),
            "Manual touch (h)": g["baseline_handling_hours"].mean(),
            "ClaimPilot touch (h)": g["claimpilot_handling_hours"].mean(),
            "Manual LAE ($)": g["baseline_lae_usd"].mean(),
            "ClaimPilot LAE ($)": g["claimpilot_lae_usd"].mean(),
            "Manual cycle (d)": g["baseline_cycle_days"].mean(),
            "ClaimPilot cycle (d)": g["claimpilot_cycle_days"].mean(),
        }
    )
    out["LAE delta ($)"] = out["ClaimPilot LAE ($)"] - out["Manual LAE ($)"]
    return out.round(2).sort_values("Claims", ascending=False).reset_index()


def lae_breakeven(df: pd.DataFrame) -> dict[str, Any]:
    """How many SIU referrals the book can carry before automation stops paying.

    Exact, not simulated: SIU time is the only term that scales with the
    referral count, so the break-even count solves in one step. It answers the
    obvious question — if ClaimPilot's expense line is higher, how much of
    that is the investigation programme rather than the automation?
    """
    costs = config.assumption("costs")
    siu_hour_cost = float(costs["siu_investigator_cost_per_hour_usd"])
    siu_hours = float(config.assumption("handling_time_hours", "claimpilot", "siu_referral_extra"))
    cost_per_referral = siu_hours * siu_hour_cost

    manual_total = float(df["baseline_lae_usd"].sum())
    # ClaimPilot's expense with the investigation programme stripped out.
    cp_ex_siu = float(
        (df["claimpilot_lae_usd"] - df["claimpilot_siu_hours"] * siu_hour_cost).sum()
    )
    headroom = manual_total - cp_ex_siu
    breakeven_n = headroom / cost_per_referral if cost_per_referral else 0.0
    actual_n = int(df["routed_to_siu"].sum())

    return {
        "cost_per_referral_usd": round(cost_per_referral, 2),
        "claimpilot_lae_excluding_siu_usd": round(cp_ex_siu, 2),
        "manual_lae_usd": round(manual_total, 2),
        "automation_headroom_usd": round(headroom, 2),
        "breakeven_referrals": int(max(breakeven_n, 0)),
        "breakeven_referral_rate": round(max(breakeven_n, 0) / len(df), 4)
        if len(df)
        else 0.0,
        "actual_referrals": actual_n,
        "actual_referral_rate": round(actual_n / len(df), 4) if len(df) else 0.0,
        "verdict": (
            "the investigation programme costs more than the automation saves"
            if actual_n > breakeven_n
            else "the automation saving covers the investigation programme"
        ),
        "note": (
            "Exact arithmetic on the assumed rates, not a simulation. The "
            "break-even rate is what a realistic auto book would sit under: "
            "this dataset's 24.7% reported-fraud rate forces a referral rate "
            "an order of magnitude above a real portfolio's."
        ),
    }


def deltas(df: pd.DataFrame) -> dict[str, Any]:
    """ClaimPilot vs. manual, as signed changes. Negative = ClaimPilot lower."""
    t = arm_totals(df)
    m, c = t["manual"], t["claimpilot"]

    def pct(new: float, old: float) -> float | None:
        return round((new - old) / old, 4) if old else None

    return {
        "handling_hours_pct": pct(c.mean_handling_hours, m.mean_handling_hours),
        "cycle_days_pct": pct(c.mean_cycle_days, m.mean_cycle_days),
        "first_contact_pct": pct(
            c.time_to_first_contact_hours, m.time_to_first_contact_hours
        ),
        "lae_per_claim_pct": pct(c.mean_lae_usd, m.mean_lae_usd),
        "lae_total_usd": round(c.total_lae_usd - m.total_lae_usd, 2),
        "leakage_total_usd": round(c.total_leakage_usd - m.total_leakage_usd, 2),
        "total_cost_usd": round(c.total_cost_usd - m.total_cost_usd, 2),
        "total_cost_pct": pct(c.total_cost_usd, m.total_cost_usd),
        "adjuster_fte_months": round(
            c.adjuster_fte_months - m.adjuster_fte_months, 1
        ),
    }


def summary(df: pd.DataFrame) -> dict[str, Any]:
    """Everything the dashboard and metrics.json need, in one dict."""
    totals = arm_totals(df)
    return {
        "version": BACKTEST_VERSION,
        "design": "paired: the ClaimPilot arm is derived from each claim's own baseline draw",
        "claims": int(len(df)),
        "arms": {k: vars(v) for k, v in totals.items()},
        "deltas": deltas(df),
        "siu": siu_hit_rate(df),
        "fast_tracked": int(df["fast_tracked"].sum()),
        "referred_to_siu": int(df["routed_to_siu"].sum()),
        "is_real": False,
        "evidence_caveat": (
            "Not evidence. The arms are paired on the same random draw, so "
            "every delta here is an arithmetic consequence of the multipliers "
            "in config/assumptions.yaml rather than a measured outcome. The "
            "only measured figures in this backtest are the routing mix and "
            "the SIU hit rate."
        ),
        "interpretation_note": str(
            config.assumption("backtest", "interpretation_note")
        ).strip(),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run() -> pd.DataFrame:
    """Build the backtest table and write it to data/sample/backtest.parquet."""
    from claimpilot.ingest import load_claims
    from claimpilot.pipeline import load_scored

    baseline = pd.read_parquet(config.SAMPLE_DIR / "baseline.parquet")
    df = build_arms(baseline, load_scored(), load_claims())
    config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(BACKTEST_PARQUET, index=False)
    print(f"Wrote {BACKTEST_PARQUET} ({df.shape[0]} rows x {df.shape[1]} cols)")
    return df


def load_backtest() -> pd.DataFrame:
    if not BACKTEST_PARQUET.exists():
        raise FileNotFoundError(
            f"{BACKTEST_PARQUET} is missing. Run `make backtest` to build it."
        )
    return pd.read_parquet(BACKTEST_PARQUET)
