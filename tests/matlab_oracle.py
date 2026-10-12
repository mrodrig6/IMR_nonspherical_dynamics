"""Evaluate expressions straight out of the legacy MATLAB sources.

The hand-expanded coefficient formulas in ``matlab_legacy/`` are the only
statement of the linearised model that exists; there is no derivation document.
Porting them is therefore a transcription problem, and transcription errors are
exactly the kind that silently survive a plausibility check.

This module removes the transcription step from the trust chain.  It reads the
``.m`` files, extracts the right-hand side of a named assignment, rewrites
MATLAB array syntax into Python, and evaluates it.  The tests then compare
:mod:`imrns.assemble.kernels` against the MATLAB text itself.

The translation handled here covers what these particular files use -- element-
wise operators, line continuations, comments, and ``hyp2f1`` -- and nothing
more.  It is a test fixture, not a MATLAB interpreter.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.special import hyp2f1

LEGACY = Path(__file__).resolve().parents[1] / "matlab_legacy" / "common"


def _strip_comments(src: str) -> str:
    """Remove ``%`` comments, respecting that ``%`` inside a string is literal."""
    out = []
    for line in src.splitlines():
        in_str = False
        cut = len(line)
        for i, ch in enumerate(line):
            if ch == "'":
                in_str = not in_str
            elif ch == "%" and not in_str:
                cut = i
                break
        out.append(line[:cut])
    return "\n".join(out)


def extract_rhs(filename: str, name: str) -> str:
    """Return the right-hand side of the assignment to ``name`` in ``filename``.

    Only the *first* assignment is returned, which is what the kernels need --
    none of the extracted coefficients is reassigned.
    """
    src = _strip_comments((LEGACY / filename).read_text())
    src = src.replace("...\n", " ")  # join continuation lines
    pattern = rf"^\s*{re.escape(name)}\s*=\s*(.*?);"
    match = re.search(pattern, src, re.MULTILINE | re.DOTALL)
    if match is None:
        raise KeyError(f"no assignment to {name!r} found in {filename}")
    return " ".join(match.group(1).split())


def to_python(expr: str) -> str:
    """Rewrite a MATLAB scalar/elementwise expression as Python."""
    expr = expr.replace(".*", "*").replace("./", "/").replace(".^", "**")
    # Remaining '^' is MATLAB's power operator (matrix power never appears here).
    expr = expr.replace("^", "**")
    # x(:) is a reshape to a column, which is a no-op for these comparisons.
    expr = expr.replace("(:)", "")
    return expr


class Indexable:
    """Wrap a value so MATLAB's ``v(i)`` call syntax resolves to it.

    The source indexes per-mode arrays as ``ep(i)``.  The tests supply one mode
    at a time, so every index maps to the same value.
    """

    def __init__(self, value):
        self.value = value

    def __call__(self, _index):
        return self.value


def evaluate(filename: str, name: str, env: dict) -> np.ndarray:
    """Extract, translate and evaluate ``name`` from ``filename`` under ``env``."""
    code = to_python(extract_rhs(filename, name))
    scope = {"hyp2f1": hyp2f1, "np": np, "__builtins__": {}}
    scope.update(env)
    try:
        return eval(code, scope)
    except Exception as exc:  # pragma: no cover - surfaces as a test failure
        raise AssertionError(f"could not evaluate {name} from {filename}:\n{code}\n{exc}") from exc
