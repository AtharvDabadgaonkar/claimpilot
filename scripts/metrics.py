"""Recompute reports/metrics.json. Entry point for `make metrics`.

Every number published in the README or the app comes from this file.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claimpilot.kpis import write_metrics  # noqa: E402

if __name__ == "__main__":
    m = write_metrics()
    print(f"Wrote reports/metrics.json ({len(json.dumps(m))} bytes)")
    print(f"  claims:            {m['dataset']['claims']}")
    print(f"  fraud rate:        {m['dataset']['reported_fraud_rate']:.1%}")
    print(f"  PII detection:     {m['pii_masking']['detection_rate']:.1%}")
    print(f"  false positives:   {m['pii_masking']['false_positive_count_on_clean_narratives']}")
    print(f"  excluded fields:   {m['governance']['fields_excluded_count']}")
    print(f"  extraction acc:    {m['extraction']['field_accuracy']:.1%}")
    print(f"  fraud AUC (CV):    {m['fraud_model']['roc_auc_cv_mean']:.3f}")
    print(f"  STP rate:          {m['triage_routing']['stp_rate']:.1%}")
    print(f"  SIU referral rate: {m['triage_routing']['siu_referral_rate']:.1%}")
    print(f"  auto-denied:       {m['triage_routing']['auto_denied_count']}")
    print(f"  reserve MAE:       ${m['reserve_model']['mae_usd']:,.0f}")
    print(f"  reserve coverage:  {m['reserve_model']['interval_coverage']:.1%}")
    print(f"  override rate:     {m['workbench']['override_rate']:.1%}")
    g = m['governance_programme']
    print(f"  fairness flagged:  {g['fairness']['tests_flagged']}/{g['fairness']['tests_assessable']} (worst {g['fairness']['worst_ratio']})")
    print(f"  model cards:       {g['model_inventory']['components']}")
    print(f"  monitoring alerts: {g['monitoring']['alerts']} / warnings {g['monitoring']['warnings']}")
    b = m['backtest']
    print(f"  backtest arms:     manual vs claimpilot ({b['claims']} claims, paired)")
    print(f"  LAE+leakage delta: {b['deltas']['total_cost_pct']:+.1%}")
    print(f"  SIU hit rate:      {b['siu']['hit_rate_precision']:.1%} (base {b['siu']['portfolio_flag_rate']:.1%})")
    print(f"  LAE breakeven:     {b['lae_breakeven']['breakeven_referral_rate']:.1%} referral rate vs actual {b['lae_breakeven']['actual_referral_rate']:.1%}")
