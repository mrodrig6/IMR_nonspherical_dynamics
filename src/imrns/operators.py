"""Finite-difference and quadrature operators on the uniform computational grid.

Three families are built here, all as dense ``xN x xN`` matrices because the
integral kernels they are combined with are dense anyway:

* :func:`difference_stencil` -- centred second-order ``d/dx``, ``d2/dx2`` and
  ``d3/dx3`` with one-sided closures at both ends.
* :func:`upwind_advection` -- the transport operator for the moving-grid term,
  biased by the sign of the local grid velocity.
* :func:`trapezoid_matrices` -- quadrature weights and the cumulative
  integration matrix used by the kappa / J / K radial kernels.

Transliterated from ``matlab_legacy/common/f_difference_stencil.m``,
``f_advective_D1_operator.m`` and ``f_trapz_integration_matrices.m``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = ["difference_stencil", "trapezoid_matrices", "upwind_advection"]


@lru_cache(maxsize=32)
def _stencils(xN: int, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if xN < 5:
        raise ValueError(f"need xN >= 5 for the third derivative, got {xN}")

    d1 = np.zeros((xN, xN))
    idx = np.arange(1, xN - 1)
    d1[idx, idx - 1] = -1.0 / (2 * h)
    d1[idx, idx + 1] = 1.0 / (2 * h)
    d1[0, 0:3] = np.array([-3.0, 4.0, -1.0]) / (2 * h)
    d1[-1, -3:] = np.array([1.0, -4.0, 3.0]) / (2 * h)

    d2 = np.zeros((xN, xN))
    d2[idx, idx - 1] = 1.0 / h**2
    d2[idx, idx] = -2.0 / h**2
    d2[idx, idx + 1] = 1.0 / h**2
    d2[0, 0:4] = np.array([2.0, -5.0, 4.0, -1.0]) / h**2
    d2[-1, -4:] = np.array([-1.0, 4.0, -5.0, 2.0]) / h**2

    # Composed rather than independently derived, so that the discrete identity
    # D3 = D1 D2 holds exactly.  The legacy code does the same.
    d3 = d1 @ d2
    return d1, d2, d3


def difference_stencil(xN: int, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(D1, D2, D3)``, second-order accurate on a uniform grid of step ``h``.

    Results are cached: the stencils depend only on ``(xN, h)``, which are fixed
    for a whole run.
    """
    d1, d2, d3 = _stencils(int(xN), float(h))
    return d1.copy(), d2.copy(), d3.copy()


@lru_cache(maxsize=32)
def _forward_d1(xN: int, h: float) -> np.ndarray:
    """Second-order forward-biased first derivative (cached: depends only on xN, h)."""
    d = np.zeros((xN, xN))
    for i in range(xN - 2):
        d[i, i : i + 3] = np.array([-3.0, 4.0, -1.0]) / (2 * h)
    d[xN - 2, xN - 3 : xN] = np.array([-1.0, 0.0, 1.0]) / (2 * h)
    d[xN - 1, xN - 3 : xN] = np.array([1.0, -4.0, 3.0]) / (2 * h)
    return d


@lru_cache(maxsize=32)
def _backward_d1(xN: int, h: float) -> np.ndarray:
    """Second-order backward-biased first derivative (cached: depends only on xN, h)."""
    d = np.zeros((xN, xN))
    d[0, 0:3] = np.array([-3.0, 4.0, -1.0]) / (2 * h)
    d[1, 0:3] = np.array([-1.0, 0.0, 1.0]) / (2 * h)
    for i in range(2, xN):
        d[i, i - 2 : i + 1] = np.array([1.0, -4.0, 3.0]) / (2 * h)
    return d


def upwind_advection(c: np.ndarray, h: float) -> np.ndarray:
    """Operator approximating ``c(x) dq/dx`` for a right-hand-side transport term.

    The equations are written as ``q_t = ... + c(x) q_x``, so characteristics run
    towards smaller ``x`` where ``c > 0``.  The stencil is therefore
    forward-biased for positive ``c`` and backward-biased for negative ``c``:

        c q_x ~ max(c, 0) D_forward q + min(c, 0) D_backward q

    This is only valid for first-derivative transport terms.  Do not reuse it
    for second or third derivatives.
    """
    c = np.asarray(c, dtype=float).ravel()
    xN = c.size
    if xN < 3:
        raise ValueError(f"need at least 3 points for the advection stencil, got {xN}")
    # The operator depends on the mode only through nothing at all -- it is the
    # same for every mode at a given timestep, so the solver builds it once and
    # passes it down.
    return (
        np.maximum(c, 0.0)[:, None] * _forward_d1(xN, float(h))
        + np.minimum(c, 0.0)[:, None] * _backward_d1(xN, float(h))
    )


def trapezoid_matrices(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trapezoidal quadrature on a strictly increasing grid ``x``.

    Returns ``(W, w, One_wT)``:

    * ``w`` -- weights for the whole-interval integral, ``int f dx ~ w @ f``.
    * ``W`` -- lower-triangular cumulative integration matrix, so that
      ``(W @ f)[i]`` approximates ``int_{x[0]}^{x[i]} f dx``.
    * ``One_wT`` -- ``ones(xN, 1) @ w.T``, used to form tail integrals as
      ``One_wT - W``.

    The legacy implementation filled ``W`` with an ``O(xN^2)`` double loop; the
    rows are built here by cumulative sums instead.
    """
    x = np.asarray(x, dtype=float).ravel()
    xN = x.size
    if xN < 2:
        raise ValueError("need at least 2 grid points")
    if not np.all(np.isfinite(x)):
        raise ValueError("grid must be finite")
    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("grid must be strictly increasing")

    w = np.zeros(xN)
    w[0] = dx[0] / 2
    if xN > 2:
        w[1:-1] = (dx[:-1] + dx[1:]) / 2
    w[-1] = dx[-1] / 2

    # Row i holds the trapezoid weights restricted to the panels [x0, xi].
    # Interior columns carry the full two-panel weight; the two endpoints of the
    # truncated interval carry half a panel each.
    W = np.zeros((xN, xN))
    interior = np.zeros(xN)
    interior[1:-1] = (dx[:-1] + dx[1:]) / 2
    for i in range(1, xN):
        W[i, 0] = dx[0] / 2
        if i > 1:
            W[i, 1:i] = interior[1:i]
        W[i, i] = dx[i - 1] / 2

    return W, w, np.ones((xN, 1)) * w[None, :]
