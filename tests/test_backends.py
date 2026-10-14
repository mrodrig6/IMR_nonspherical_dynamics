"""All three backends must produce the same numbers.

The physics lives in one module (:mod:`imrns.assemble.kernels`); the backends
differ only in how the block is built and solved.  These tests pin that down.
"""

from __future__ import annotations

import numpy as np
import pytest

from imrns import grid as g
from imrns.assemble import numpy_backend as nb
from imrns.backends import available_backends
from imrns.layout import build_layout
from imrns.operators import difference_stencil, trapezoid_matrices
from imrns.solver import solve_perturbation

XN = 32
STATE = (1.23, 0.37, -0.19, 1.0)
GROUPS = (1 / 50.0, 1 / 35.0, 1 / 45.0)
ALPHA = 0.13


def _ops():
    x = g.computational_grid(XN)
    W, w, One_wT = trapezoid_matrices(x)
    return x, g.grid_spacing(x), (*difference_stencil(XN, g.grid_spacing(x)), W, w, One_wT)


def _numpy_lhs(n, dt, closure):
    """Assemble the left-hand side the reference backend produces, unscaled."""
    x, _h, ops = _ops()
    layout = build_layout(n, XN, "me", closure, False)
    loc = layout.local(0)
    metrics = g.grid_metrics(x, STATE[0], STATE[1], STATE[3], 5.0)
    hyp = nb.hyp2f1_values(float(n), metrics.sr)
    A = nb.assemble_block(n, loc, metrics, STATE, GROUPS, ALPHA, "me", closure, False, ops, hyp)
    if closure != "irr":
        nb.ep_feedback_rows(A, n, loc, metrics, STATE, GROUPS, ALPHA, "me", ops, hyp)
        nb.add_moving_grid_terms(A, loc, metrics, "me")
    lhs = -dt * A
    lhs[np.diag_indices(lhs.shape[0])] += 1.0
    rhs = np.zeros(lhs.shape[0])
    if closure != "irr":
        nb.apply_boundary_conditions(lhs, rhs, n, loc, metrics, STATE, GROUPS, ALPHA,
                                     "me", False, ops, 0.0, 0.0)
    return lhs


@pytest.mark.slow
@pytest.mark.skipif("numba" not in available_backends(), reason="numba not installed")
@pytest.mark.parametrize("closure", ["rot", "irr"])
def test_numba_matches_numpy_end_to_end(closure):
    t = np.linspace(0.0, 1.5, 120)
    kw = dict(modes=[2, 5], t=t, R=1 + 0.1 * np.sin(2 * t), Rd=0.2 * np.cos(2 * t),
              Rdd=-0.4 * np.sin(2 * t), Ca=50.0, Re=35.0, We=45.0, alpha=ALPHA,
              xN=XN, ep0=[0.01, -0.02], closure=closure)
    a = solve_perturbation(backend="numpy", **kw)
    b = solve_perturbation(backend="numba", **kw)
    np.testing.assert_allclose(b.ep, a.ep, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(b.T, a.T, rtol=1e-12, atol=1e-15)


@pytest.mark.slow
@pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")
@pytest.mark.parametrize("n,closure", [(5, "rot"), (8, "rot"), (5, "irr")])
def test_jax_assembles_the_same_operator(n, closure):
    import jax.numpy as jnp

    from imrns.assemble import jax_backend as jb

    dt = 1e-3
    x, h, _ = _ops()
    lhs_jax, _ = jb.assemble_lhs(
        float(n), XN, closure, 1.0, dt, *STATE[:3], STATE[3], 5.0, *GROUPS, ALPHA,
        jnp.asarray(x), jb.make_ops(XN, h),
    )
    np.testing.assert_allclose(np.asarray(lhs_jax), _numpy_lhs(n, dt, closure),
                               rtol=1e-11, atol=1e-13)


@pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")
def test_jax_hyp2f1_matches_scipy_and_its_derivative():
    import jax
    import jax.numpy as jnp

    from imrns.assemble import jax_backend as jb
    from imrns.special import d_hyp2f1_dz, hyp2f1

    a, b, c = -1 / 3, 4 + 8 / 3, 5 + 8 / 3
    z = jnp.array([0.2, -0.5, 0.75])
    np.testing.assert_allclose(np.asarray(jb.hyp2f1(a, b, c, z)), hyp2f1(a, b, c, np.asarray(z)),
                               rtol=1e-13)
    grad = jax.grad(lambda zz: jb.hyp2f1(a, b, c, zz).sum())(z)
    np.testing.assert_allclose(np.asarray(grad), d_hyp2f1_dz(a, b, c, np.asarray(z)), rtol=1e-10)
