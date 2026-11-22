"""Generate the simulated layers. Entry point for `make simulate`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claimpilot.simulate import run  # noqa: E402

if __name__ == "__main__":
    print("Generating simulated tables (narratives, policies, deadlines, baseline)...")
    run()
