"""Download and process the source dataset. Entry point for `make data`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claimpilot.ingest import run  # noqa: E402

if __name__ == "__main__":
    run(force_download="--force" in sys.argv)
