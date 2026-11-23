"""Score the whole portfolio. Entry point for `make score`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claimpilot.pipeline import run  # noqa: E402

if __name__ == "__main__":
    print("Scoring portfolio (coverage -> fraud -> triage -> routing)...")
    run()
