"""The moving-grid (ALE) term: is it required, and is it implemented correctly?

The grid ``r = R(t) g(x)`` moves with the bubble wall, so a node of constant
``x`` is at neither a fixed physical location nor a fixed material point.  The
time-stepper advances nodal values, i.e. ``d/dt|_x``, while the governing
equations are written with ``d/dt|_r``.  The term reconciling them is

    dphi/dt|_x = dphi/dt|_r + a phi_x,
    a(x, t) = (Rdot/R) (1-x)^2 / (2L) [1 + L(1+x)/(1-x)]

which is :func:`imrns.grid.grid_velocity`.  These tests pin that it is exactly
right, that nothing is missing from it, and that removing it breaks the scheme.

See ``docs/moving-grid.md`` for the derivation and for why a stale comment in
the MATLAB source made this look like an unmotivated fudge term.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from imrns.grid import computational_grid, grid_spacing, grid_velocity, radius
from imrns.operators import _backward_d1, _forward_d1, difference_stencil

L = 5.0
R0, RD0 = 1.37, 0.41


def _g(x):
    return 1.0 + L * (1.0 + x) / (1.0 - x)


@pytest.mark.parametrize("x0", [-0.9, -0.5, 0.0, 0.5, 0.9])
def test_grid_velocity_is_minus_the_drift_of_a_fixed_physical_point(x0):
    """``a = -dx/dt|_r``.

    Hold a physical radius fixed, perturb R, solve for the computational
    coordinate, and difference.  This is the kinematic definition of the term,
    checked without reference to the algebra it was derived from.
    """
    r_fixed = R0 * _g(x0)
    eps = 1e-7

    def x_of_R(R):
        return brentq(lambda x: R * _g(x) - r_fixed, -1 + 1e-14, 1 - 1e-14,
                      xtol=1e-15, rtol=1e-15)

    dxdt = (x_of_R(R0 + RD0 * eps) - x_of_R(R0 - RD0 * eps)) / (2 * eps)
    a = float(grid_velocity(np.array([x0]), R0, RD0, L)[0])
    assert a == pytest.approx(-dxdt, rel=1e-6)


def _manufactured(r, t):
    return np.sin(0.7 * r) * np.exp(-0.3 * t) / (1.0 + 0.1 * r**2)


def _ale_residual(xN, coefficient=1.0):
    """Relative residual of the ALE identity on a manufactured field."""
    t0, eps = 0.4, 1e-6
    x = computational_grid(xN)
    D1, _, _ = difference_stencil(xN, grid_spacing(x))

    def nodal(t):
        return _manufactured(radius(x, R0 + RD0 * (t - t0), L), t)

    lhs = (nodal(t0 + eps) - nodal(t0 - eps)) / (2 * eps)          # dphi/dt|_x
    r = radius(x, R0, L)
    rhs = -0.3 * _manufactured(r, t0) + coefficient * grid_velocity(x, R0, RD0, L) * (
        D1 @ _manufactured(r, t0)
    )
    window = (r > 1.5) & (r < 40.0)
    return np.abs(lhs - rhs)[window].max() / np.abs(lhs)[window].max()


def test_ale_identity_residual_converges_at_second_order():
    """Nothing is missing from the term.

    The residual is pure D1 truncation error: it falls by about four per grid
    doubling and keeps going.  A missing term would plateau at a nonzero value
    instead, which is the whole point of running this as a refinement study
    rather than at a single resolution.
    """
    residuals = [_ale_residual(xN) for xN in (301, 601, 1201, 2401)]
    for coarse, fine in zip(residuals[:-1], residuals[1:]):
        assert 3.0 < coarse / fine < 5.0
    assert residuals[-1] < 2e-4


def test_the_coefficient_is_pinned_to_one():
    """Coefficient 2 -- the value a reader might infer from the stale docstring
    in f_modify_A_bulk_map.m -- is wrong by order one, not marginally worse."""
    assert _ale_residual(1201, coefficient=1.0) < 1e-3
    assert _ale_residual(1201, coefficient=2.0) > 0.5
    assert _ale_residual(1201, coefficient=0.0) > 0.5


def test_a_field_frozen_in_physical_space_stays_frozen():
    """The term's actual job, and what breaks without it.

    Set ``dT/dt|_r = 0`` so the T row reduces to ``dT/dt|_x = a T_x``, and
    integrate while the wall moves.  A bump fixed in physical space must stay
    where it is.  Removing the term pins the field to grid nodes that are
    themselves sliding through physical space, which is the error it exists to
    prevent.
    """
    bump = lambda r: np.exp(-(((r - 6.0) / 3.0) ** 2))  # noqa: E731
    xN = 801
    x = computational_grid(xN)
    h = float(grid_spacing(x))
    Df, Db = _forward_d1(xN, h), _backward_d1(xN, h)
    R = lambda t: R0 * (1.0 + 0.25 * np.sin(1.3 * t))   # noqa: E731
    Rd = lambda t: R0 * 0.25 * 1.3 * np.cos(1.3 * t)    # noqa: E731

    def rhs(t, u):
        a = grid_velocity(x, R(t), Rd(t), L)
        return np.maximum(a, 0.0) * (Df @ u) + np.minimum(a, 0.0) * (Db @ u)

    T = bump(radius(x, R(0.0), L))
    initial = T.copy()
    dt, n_steps = 1e-3, 2000
    for k in range(n_steps):
        tk = k * dt
        k1 = rhs(tk, T)
        k2 = rhs(tk + dt, T + dt * k1)
        T = T + 0.5 * dt * (k1 + k2)

    t_end = n_steps * dt
    r_end = radius(x, R(t_end), L)
    exact = bump(r_end)
    window = (r_end > 1.7) & (r_end < 30.0)

    assert abs(R(t_end) / R(0.0) - 1.0) > 0.1, "the wall must actually move"
    with_term = np.abs(T - exact)[window].max()
    without_term = np.abs(initial - exact)[window].max()
    assert with_term < 1e-3
    assert without_term > 0.1
    assert without_term / with_term > 100


def test_grid_velocity_boundary_behaviour():
    """No spurious transport at the far field; correct magnitude at the wall."""
    assert float(grid_velocity(np.array([-1.0]), R0, RD0, L)[0]) == pytest.approx(
        2 * RD0 / (R0 * L)
    )
    far = grid_velocity(np.array([1.0 - 1e-8]), R0, RD0, L)[0]
    assert abs(far) < 1e-8

    x = computational_grid(64)
    assert np.all(grid_velocity(x, R0, 0.4, L) > 0), "expansion drives a > 0"
    assert np.all(grid_velocity(x, R0, -0.4, L) < 0), "collapse drives a < 0"
    assert np.allclose(grid_velocity(x, R0, 0.0, L), 0.0), "a static wall needs no term"


def test_upwind_direction_follows_the_characteristics():
    """For ``phi_t = a phi_x`` the characteristics are ``dx/dt = -a``.

    So a > 0 must reach towards larger x (forward bias) and a < 0 towards
    smaller x.  Getting this backwards gives an unconditionally unstable
    scheme, so it is worth pinning explicitly.
    """
    from imrns.operators import upwind_advection

    xN, h = 20, 0.1
    forward = upwind_advection(np.ones(xN), h)
    backward = upwind_advection(-np.ones(xN), h)
    assert forward[5, 6] != 0.0 and forward[5, 4] == 0.0
    assert backward[15, 14] != 0.0 and backward[15, 16] == 0.0


def test_stale_docstring_describes_the_same_equation_in_other_variables():
    """``f_modify_A_bulk_map.m``'s docstring is correct maths, wrong variables.

    It states ``V_t = Y + (a_t - a a_x) T_x + 2a V_x - a^2 T_xx`` while the code
    adds only ``a V_x``.  Both are right: the docstring is for
    ``W = dT/dt|_x``, the code for ``V = dT/dt|_r``.  Substituting
    ``V = W - a T_x`` into the implemented row reproduces the docstring exactly,
    so the difference is identically zero.

    This is the test that answers "is the term a fudge" -- it is not, it is one
    formulation of the same kinematics.
    """
    sympy = pytest.importorskip("sympy")

    x, t = sympy.symbols("x t")
    a = sympy.Function("a")(x, t)
    T = sympy.Function("T")(x, t)
    Y = sympy.Function("Y")(x, t)

    W = sympy.diff(T, t)              # the docstring's variable, dT/dt|_x
    V = W - a * sympy.diff(T, x)      # the implemented variable, dT/dt|_r

    implemented = sympy.expand(sympy.diff(V, t) - (Y + a * sympy.diff(V, x)))
    docstring = sympy.expand(
        sympy.diff(W, t)
        - (
            Y
            + (sympy.diff(a, t) - a * sympy.diff(a, x)) * sympy.diff(T, x)
            + 2 * a * sympy.diff(W, x)
            - a**2 * sympy.diff(T, x, 2)
        )
    )
    assert sympy.simplify(implemented - docstring) == 0
