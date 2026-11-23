"""End-to-end scoring pipeline.

Runs coverage -> fraud model -> triage -> routing over the whole portfolio and
writes the result to a committed Parquet file. The deployed app reads that file
instead of recomputing, so page loads stay fast and the demo never depends on
training at runtime.
"""

from __future__ import annotations

import pandas as pd

from claimpilot import config
from claimpilot.coverage import check_coverage_batch
from claimpilot.fraud_model import predict_proba
from claimpilot.routing import route_batch
from claimpilot.triage import triage_batch

SCORED_PARQUET = config.SAMPLE_DIR / "scored.parquet"


def score_portfolio(
    claims: pd.DataFrame | None = None, policies: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Coverage, triage and routing for every claim, as one table."""
    from claimpilot.ingest import load_claims

    claims = load_claims() if claims is None else claims
    if policies is None:
        policies = pd.read_parquet(config.POLICIES_PARQUET)

    coverage = check_coverage_batch(claims, policies)
    probabilities = predict_proba(claims)
    triage = triage_batch(claims, coverage, probabilities)
    routing = route_batch(triage)

    scored = (
        triage.merge(
            coverage.drop(columns=["coverage_status"]), on="claim_id", how="left"
        )
        .merge(routing.drop(columns=["claim_id"]).assign(claim_id=triage["claim_id"]),
               on="claim_id", how="left")
    )
    return scored


def run() -> pd.DataFrame:
    scored = score_portfolio()
    config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(SCORED_PARQUET, index=False)
    size_kb = SCORED_PARQUET.stat().st_size / 1024
    print(
        f"Scored {len(scored)} claims -> "
        f"{SCORED_PARQUET.relative_to(config.ROOT)} ({size_kb:.0f} KB)"
    )
    print(f"  STP auto-approved: {scored['auto_approved'].mean():.1%}")
    print(f"  SIU referrals:     {(scored['destination'] == 'siu_queue').mean():.1%}")
    return scored


def load_scored() -> pd.DataFrame:
    if not SCORED_PARQUET.exists():
        raise FileNotFoundError(
            f"{SCORED_PARQUET} is missing. Run `make score` to build it."
        )
    return pd.read_parquet(SCORED_PARQUET)
