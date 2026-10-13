"""Radial coordinate map between the physical domain and the solver domain.

The perturbation equations live on ``r in [R(t), inf)``.  That is mapped onto a
finite computational interval ``x in [-1, 1)`` by

    r(x) = R (1 + L (1 + x) / (1 - x)),

so ``x = -1`` is the bubble wall ``r = R`` and ``x -> 1`` is the far field.  The
stretching parameter ``L`` sets how much of the grid sits near the wall: the
point ``x = 0`` lands at ``r = R(1 + L)``.

Because the map depends on ``R(t)``, the grid moves.  Derivatives with respect
to ``r`` therefore pick up chain-rule factors, and the time derivative at fixed
``x`` differs from the one at fixed ``r`` by a grid velocity.  This module
supplies all of those metric quantities.

Transliterated from ``matlab_legacy/common/f_r.m``, ``f_ds.m``,
``f_diff_var_change.m``, ``f_diff2_var_change.m``, ``f_diff3_var_change.m`` and
``f_grid_velocity.m``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The grid stops just short of x = 1 because the map is singular there.
DEFAULT_WALL_GAP = 1e-4


def computational_grid(xN: int, wall_gap: float = DEFAULT_WALL_GAP) -> np.ndarray:
    """Return the uniform computational grid ``x`` with ``xN`` points.

    The last point is pulled back to ``1 - wall_gap`` to keep ``r`` finite.
    """
    if xN < 5:
        raise ValueError(f"xN must be at least 5 for the third-derivative stencil, got {xN}")
    return np.linspace(-1.0, 1.0 - wall_gap, xN)


def grid_spacing(x: np.ndarray) -> float:
    """Uniform spacing of a grid produced by :func:`computational_grid`.

    The legacy code mixed two values here: the stencils were built with
    ``(2 - wall_gap)/(xN - 1)`` while the advection operator used ``2/(xN - 1)``.
    Both operators now take this single value.
    """
    return float(x[1] - x[0])


def radius(x: np.ndarray, R: float, L: float) -> np.ndarray:
    """Physical radius ``r`` at computational coordinate ``x``."""
    return R * ((1.0 + x) / (1.0 - x) * L + 1.0)


def drdx(x: np.ndarray, R: float, L: float) -> np.ndarray:
    """``dr/dx``, the radial integration element for quadrature in ``x``."""
    return 2.0 * R * L / (1.0 - x) ** 2


def dxdr(x: np.ndarray, R: float, L: float) -> np.ndarray:
    """``dx/dr``: the factor converting ``d/dx`` into ``d/dr``."""
    return (1.0 - x) ** 2 / (2.0 * R * L)


def d2xdr2(x: np.ndarray, R: float, L: float) -> tuple[np.ndarray, np.ndarray]:
    """Chain-rule coefficients for the second radial derivative.

    Returns ``(c2, c1)`` such that ``d2/dr2 = c2 d2/dx2 + c1 d/dx``.
    """
    c1 = -((1.0 - x) ** 3) / (2.0 * R**2 * L**2)
    c2 = (1.0 - x) ** 4 / (4.0 * R**2 * L**2)
    return c2, c1


def d3xdr3(x: np.ndarray, R: float, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chain-rule coefficients for the third radial derivative.

    Returns ``(c3, c2, c1)`` such that ``d3/dr3 = c3 d3/dx3 + c2 d2/dx2 + c1 d/dx``.
    """
    c1 = 3.0 * (1.0 - x) ** 4 / (4.0 * R**3 * L**3)
    c2 = -3.0 * (1.0 - x) ** 5 / (4.0 * R**3 * L**3)
    c3 = (1.0 - x) ** 6 / (8.0 * R**3 * L**3)
    return c3, c2, c1


def grid_velocity(x: np.ndarray, R: float, Rd: float, L: float) -> np.ndarray:
    """Velocity of the moving grid: the speed physical points drift through it.

    A node of constant ``x`` sits at ``r = R(t) g(x)``, so it is at neither a
    fixed physical location nor a fixed material point.  The time-stepper
    advances nodal values -- ``d/dt|_x`` -- while the governing equations are
    written with ``d/dt|_r``.  The two are related, for *any* field, by the
    Arbitrary Lagrangian-Eulerian identity

        dphi/dt|_x = dphi/dt|_r + a dphi/dx

    and this function returns ``a``.  Equivalently ``a = -(dx/dt)|_r``: the rate
    at which a fixed physical point drifts through the computational grid,
    signed to act as a transport velocity in the right-hand-side form
    ``phi_t = ... + a phi_x``.

    This is kinematics, with no modelling content -- it is not an empirical
    correction, and it is not optional: without it the scheme integrates the
    wrong equation whenever the wall moves.  ``docs/moving-grid.md`` gives the
    three-line derivation and the numerical evidence; ``tests/test_moving_grid.py``
    pins it.
    """
    return Rd / R * (1.0 - x) ** 2 / (2.0 * L) * (1.0 + L * (1.0 + x) / (1.0 - x))


def material_radius(r: np.ndarray, R: float, Req: float) -> np.ndarray:
    """Reference (undeformed) radius ``r0`` of the material now sitting at ``r``.

    Incompressibility of the surrounding medium gives
    ``r0^3 = r^3 - R^3 + Req^3``.  The cube root is taken sign-preserving so the
    expression stays real if the argument dips negative near the wall.
    """
    arg = r**3 - R**3 + Req**3
    return np.sign(arg) * np.abs(arg) ** (1.0 / 3.0)


def volume_change(R: float, Req: float) -> float:
    """``Req^3 ((R/Req)^3 - 1)``: the volume the bubble has displaced.

    Named ``vs`` in the legacy code, where it appears throughout the elastic
    stress coefficients.
    """
    return Req**3 * ((R / Req) ** 3 - 1.0)


@dataclass(frozen=True)
class GridMetrics:
    """All grid quantities needed to assemble the operator at one instant.

    Built once per timestep by :func:`grid_metrics`.
    """

    x: np.ndarray
    h: float
    r: np.ndarray
    drdx: np.ndarray
    dxdr: np.ndarray
    d2xdr2_2: np.ndarray
    d2xdr2_1: np.ndarray
    d3xdr3_3: np.ndarray
    d3xdr3_2: np.ndarray
    d3xdr3_1: np.ndarray
    a: np.ndarray
    rs: np.ndarray
    vs: float
    sr: float


def grid_metrics(x: np.ndarray, R: float, Rd: float, Req: float, L: float) -> GridMetrics:
    """Evaluate every metric quantity on grid ``x`` for the state ``(R, Rd)``."""
    r = radius(x, R, L)
    c2, c1 = d2xdr2(x, R, L)
    e3, e2, e1 = d3xdr3(x, R, L)
    return GridMetrics(
        x=x,
        h=grid_spacing(x),
        r=r,
        drdx=drdx(x, R, L),
        dxdr=dxdr(x, R, L),
        d2xdr2_2=c2,
        d2xdr2_1=c1,
        d3xdr3_3=e3,
        d3xdr3_2=e2,
        d3xdr3_1=e1,
        a=grid_velocity(x, R, Rd, L),
        rs=material_radius(r, R, Req),
        vs=volume_change(R, Req),
        sr=R / Req,
    )
