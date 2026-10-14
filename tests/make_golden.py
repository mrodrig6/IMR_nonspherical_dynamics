"""Regenerate the golden trajectories used by test_regression.py.

Run this only when a numerical change is intended, and record why in the commit
message.  From the repository root:

    PYTHONPATH=src python tests/make_golden.py
"""

from __future__ import annotations

import numpy as np

from test_regression import CASES, GOLDEN, run


def main() -> None:
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for name, spec in sorted(CASES.items()):
        result = run(spec)
        data[f"{name}__ep"] = result.ep
        data[f"{name}__T"] = result.T
        print(f"{name:16s} ep peak {np.abs(result.ep).max():.6e}")
    np.savez_compressed(GOLDEN, **data)
    print(f"wrote {GOLDEN}")


if __name__ == "__main__":
    main()
