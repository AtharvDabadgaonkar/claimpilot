"""Train and persist the models. Entry point for `make models`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claimpilot.fraud_model import train  # noqa: E402
from claimpilot.reserve import train as train_reserve  # noqa: E402

if __name__ == "__main__":
    print("Training fraud model...")
    _, metrics = train()
    m = metrics.to_dict()
    print(f"  ROC AUC (test):   {m['roc_auc_test']:.3f}")
    print(f"  ROC AUC (CV):     {m['roc_auc_cv_mean']:.3f} +/- {m['roc_auc_cv_std']:.3f}")
    print(f"  Avg precision:    {m['average_precision_test']:.3f}")
    print(f"  Threshold:        {m['threshold']}")
    print(f"  Precision@thr:    {m['precision_at_threshold']:.3f}")
    print(f"  Recall@thr:       {m['recall_at_threshold']:.3f}")
    print(f"  Flag rate:        {m['flag_rate_at_threshold']:.1%} (base rate {m['base_rate']:.1%})")

    print("\nTraining reserve model...")
    _, rmetrics = train_reserve()
    r = rmetrics.to_dict()
    print(f"  MAE:              ${r['mae_usd']:,.0f}")
    print(f"  Median abs error: ${r['median_absolute_error_usd']:,.0f}")
    print(f"  R2:               {r['r2']:.3f}")
    print(f"  Interval coverage:{r['interval_coverage']:.1%} (target 80%)")
    print(f"  Mean range width: ${r['mean_interval_width_usd']:,.0f}")
