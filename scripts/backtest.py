"""Run the ClaimPilot vs. manual backtest. Entry point for `make backtest`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claimpilot.backtest import run, summary  # noqa: E402

if __name__ == "__main__":
    df = run()
    s = summary(df)
    m, c = s["arms"]["manual"], s["arms"]["claimpilot"]
    print(f"  arms:              manual vs claimpilot ({s['claims']} claims, paired)")
    print(f"  touch time (h):    {m['mean_handling_hours']} -> {c['mean_handling_hours']}")
    print(f"  cycle time (d):    {m['mean_cycle_days']} -> {c['mean_cycle_days']}")
    print(f"  LAE per claim:     ${m['mean_lae_usd']:,.0f} -> ${c['mean_lae_usd']:,.0f}")
    print(f"  leakage total:     ${m['total_leakage_usd']:,.0f} -> ${c['total_leakage_usd']:,.0f}")
    print(f"  LAE + leakage:     ${m['total_cost_usd']:,.0f} -> ${c['total_cost_usd']:,.0f}")
    print(f"  SIU hit rate:      {s['siu']['hit_rate_precision']:.1%} vs base rate {s['siu']['portfolio_flag_rate']:.1%}")
