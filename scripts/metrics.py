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
