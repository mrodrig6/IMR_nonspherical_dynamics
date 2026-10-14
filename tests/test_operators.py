"""Convergence and identities of the discrete operators."""

from __future__ import annotations

import numpy as np
import pytest

from imrns.grid import (computational_grid, d2xdr2, d3xdr3, drdx, dxdr, grid_metrics,
                        grid_spacing, material_radius, radius)
from imrns.operators import difference_stencil, trapezoid_matrices, upwind_advection


@pytest.mark.parametrize("order,slack", [(1, 2), (2, 3), (3, 4)])
def test_difference_stencils_are_second_order(order, slack):
    """Halving the grid spacing should cut the interior error by about four."""
    errors = []
    for xN in (81, 161, 321):
        x = np.linspace(-1.0, 1.0, xN)
        h = x[1] - x[0]
        D = difference_stencil(xN, h)[order - 1]
        exact = {1: 3 * np.cos(3 * x), 2: -9 * np.sin(3 * x), 3: -27 * np.cos(3 * x)}[order]
        errors.append(np.abs(D @ np.sin(3 * x) - exact)[slack:-slack].max())
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert 3.5 < coarse / fine < 4.5


def test_third_derivative_is_the_composition_of_the_first_two():
    """D3 is built as D1 @ D2 so the discrete identity holds exactly."""
    x = np.linspace(-1.0, 1.0, 64)
    D1, D2, D3 = difference_stencil(64, x[1] - x[0])
    np.testing.assert_allclose(D3, D1 @ D2, rtol=1e-12)


def test_quadrature_is_exact_for_linear_functions():
    x = np.linspace(-1.0, 1.0, 51)
    _W, w, _ = trapezoid_matrices(x)
    np.testing.assert_allclose(w @ (2.0 * x + 3.0), 6.0, rtol=1e-12)


def test_cumulative_and_tail_quadrature_agree_with_the_antiderivative():
    x = np.linspace(-1.0, 1.0, 801)
    W, w, One_wT = trapezoid_matrices(x)
    f = np.exp(x)
    assert np.abs(W @ f - (np.exp(x) - np.exp(-1))).max() < 1e-5
    assert np.abs((One_wT - W) @ f - (np.e - np.exp(x))).max() < 1e-5
    np.testing.assert_allclose(One_wT[0], w, rtol=1e-14)


def test_upwind_operator_biases_with_the_sign_of_the_velocity():
    xN, h = 20, 0.1
    forward = upwind_advection(np.ones(xN), h)
    backward = upwind_advection(-np.ones(xN), h)
    # A forward-biased row reaches to the right of the diagonal, and vice versa.
    assert forward[5, 6] != 0 and forward[5, 4] == 0
    assert backward[15, 14] != 0 and backward[15, 16] == 0
    np.testing.assert_allclose(upwind_advection(np.zeros(xN), h), 0.0)


def test_coordinate_map_endpoints_and_inverse_metric():
    x = computational_grid(64)
    R, L = 1.3, 5.0
    r = radius(x, R, L)
    assert r[0] == pytest.approx(R)
    assert r[-1] > 1e3 * R
    # dx/dr and dr/dx are reciprocals.
    np.testing.assert_allclose(dxdr(x, R, L) * drdx(x, R, L), 1.0, rtol=1e-14)


def test_chain_rule_coefficients_match_numerical_derivatives():
    """The analytic d2x/dr2 and d3x/dr3 coefficients reproduce finite differences."""
    x = computational_grid(2001)
    R, L = 1.3, 5.0
    h = grid_spacing(x)
    r = radius(x, R, L)
    D1, D2, D3 = difference_stencil(x.size, h)
    f = 1.0 / (1.0 + r)  # smooth, decaying, with known derivatives in r

    c2, c1 = d2xdr2(x, R, L)
    e3, e2, e1 = d3xdr3(x, R, L)
    got2 = c2 * (D2 @ f) + c1 * (D1 @ f)
    got3 = e3 * (D3 @ f) + e2 * (D2 @ f) + e1 * (D1 @ f)
    sl = slice(5, 400)  # near the wall, where the map is well resolved
    np.testing.assert_allclose(got2[sl], 2.0 / (1.0 + r[sl]) ** 3, rtol=2e-3)
    np.testing.assert_allclose(got3[sl], -6.0 / (1.0 + r[sl]) ** 4, rtol=2e-2)


def test_material_radius_inverts_incompressibility():
    """r0^3 = r^3 - R^3 + Req^3 must hold wherever it is evaluated."""
    x = computational_grid(64)
    R, Req, L = 1.4, 1.0, 5.0
    r = radius(x, R, L)
    rs = material_radius(r, R, Req)
    np.testing.assert_allclose(rs**3, r**3 - R**3 + Req**3, rtol=1e-12)


def test_grid_metrics_bundle_is_self_consistent():
    x = computational_grid(32)
    m = grid_metrics(x, 1.4, 0.3, 1.0, 5.0)
    np.testing.assert_allclose(m.r, radius(x, 1.4, 5.0), rtol=1e-14)
    np.testing.assert_allclose(m.dxdr * m.drdx, 1.0, rtol=1e-14)
    assert m.sr == pytest.approx(1.4)
    assert m.vs == pytest.approx(1.4**3 - 1.0)


def test_one_grid_spacing_is_used_everywhere():
    """The stencils and the advection operator share a single h.

    The MATLAB code built the derivative stencils with (2 - 1e-4)/(xN-1) and the
    advection operator with 2/(xN-1), so the two families were inconsistent on
    the same grid.
    """
    x = computational_grid(64)
    h = grid_spacing(x)
    assert h == pytest.approx((x[-1] - x[0]) / 63)
    assert h != pytest.approx(2.0 / 63, rel=1e-9)
