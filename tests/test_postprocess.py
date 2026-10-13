"""Field reconstruction and rendering.

The field reconstruction is checked against results derivable independently of
the code: the exact incompressible base solution, the known nodal structure of
each spherical harmonic, and the limits in which measures must agree.  Nothing
here compares against the MATLAB, because the MATLAB visualiser cannot be run.
"""

from __future__ import annotations

import numpy as np
import pytest

from imrns.postprocess.fields import (build_modal_field, harmonics,
                                      rate_of_strain_components, strain_components,
                                      stress_components, tensor_scalar)
from imrns.solver import solve_perturbation

L, RREF = 5.0, 1.0


def _run(modes=5, ep0=0.02, moving=True, xN=96, n_steps=80, closure="rot"):
    t = np.linspace(0.0, 1.0, n_steps)
    if moving:
        R, Rd, Rdd = 1 + 0.3 * np.sin(2 * t), 0.6 * np.cos(2 * t), -1.2 * np.sin(2 * t)
    else:
        R, Rd, Rdd = np.ones_like(t), np.zeros_like(t), np.zeros_like(t)
    return solve_perturbation(modes=modes, t=t, R=R, Rd=Rd, Rdd=Rdd, Ca=50.0, Re=35.0,
                              We=45.0, alpha=0.1, xN=xN, ep0=ep0, closure=closure)


def _grid(field, n_r=60, n_th=24, r_max=10.0):
    """A plotting grid, already broadcast so expected values match shapes."""
    r = np.linspace(field.R * 1.05, r_max, n_r)[:, None]
    th = np.linspace(0.05, np.pi - 0.05, n_th)[None, :]
    return np.broadcast_to(r, (n_r, n_th)).copy(), np.broadcast_to(th, (n_r, n_th)).copy()


# --------------------------------------------------------------------------
# Spherical harmonics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n,closed_form", [
    (0, lambda t: np.ones_like(t)),
    (1, np.cos),
    (2, lambda t: 0.5 * (3 * np.cos(t) ** 2 - 1)),
    (3, lambda t: 0.5 * (5 * np.cos(t) ** 3 - 3 * np.cos(t))),
])
def test_axisymmetric_harmonics_match_closed_forms(n, closed_form):
    th = np.linspace(0.01, np.pi - 0.01, 21)
    norm = np.sqrt((2 * n + 1) / (4 * np.pi))
    np.testing.assert_allclose(harmonics(n, th)[0], norm * closed_form(th), atol=1e-14)


@pytest.mark.parametrize("n", [2, 5, 8])
def test_harmonic_theta_derivative(n):
    th = np.linspace(0.2, np.pi - 0.2, 15)
    h = 1e-6
    fd = (harmonics(n, th + h)[0] - harmonics(n, th - h)[0]) / (2 * h)
    np.testing.assert_allclose(harmonics(n, th)[1], fd, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("n", [1, 2, 4, 7])
def test_harmonics_are_orthonormal(n):
    """``int |Y|^2 dOmega = 1`` -- pins the normalisation constant."""
    th = np.linspace(0.0, np.pi, 20001)
    Y = harmonics(n, th)[0]
    assert np.trapezoid(Y**2 * np.sin(th), th) * 2 * np.pi == pytest.approx(1.0, rel=1e-6)


def test_sectoral_harmonics_differentiate_in_azimuth():
    th = np.linspace(0.3, np.pi - 0.3, 11)
    az = np.full_like(th, 0.7)
    h = 1e-6
    fd = (harmonics(3, th, az + h, mode="sectoral")[0]
          - harmonics(3, th, az - h, mode="sectoral")[0]) / (2 * h)
    np.testing.assert_allclose(harmonics(3, th, az, mode="sectoral")[3], fd, rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------
# Field reconstruction
# --------------------------------------------------------------------------


def test_unperturbed_strain_is_the_exact_incompressible_solution():
    """With no perturbation the field must be the analytic spherical result.

    Incompressibility gives ``u = r - rho`` with ``rho^3 = r^3 - R^3 + Rref^3``,
    hence ``e_rr = (1 - r^4/rho^4)/2`` and ``e_tt = e_pp = (1 - rho^2/r^2)/2``.
    This exercises the base displacement, the gradient formulas and the Almansi
    construction at once.
    """
    field = build_modal_field(_run(ep0=0.0), 60, L, RREF)
    assert np.abs(field.A).max() == 0.0 and np.abs(field.B).max() == 0.0

    r, th = _grid(field)
    err, ett, epp, ert, erp, etp = strain_components(field, r, th, Rref=RREF)
    rho = np.cbrt(r**3 - field.R**3 + RREF**3)

    np.testing.assert_allclose(err, 0.5 * (1 - r**4 / rho**4), atol=1e-13)
    np.testing.assert_allclose(ett, 0.5 * (1 - rho**2 / r**2), atol=1e-13)
    np.testing.assert_allclose(epp, ett, atol=1e-15)
    assert np.abs(ert).max() == 0.0


def test_unperturbed_rate_of_strain_is_the_exact_base_flow():
    """``v_r = R^2 Rdot / r^2`` gives ``D_rr = -2 R^2 Rdot / r^3`` and a zero trace."""
    field = build_modal_field(_run(ep0=0.0), 60, L, RREF, with_rates=True)
    r, th = _grid(field)
    drr, dtt, dpp, drt, _, _ = rate_of_strain_components(field, r, th)
    flux = field.R**2 * field.Rdot
    np.testing.assert_allclose(drr, -2 * flux / r**3, atol=1e-14)
    np.testing.assert_allclose(dtt, flux / r**3, atol=1e-14)
    np.testing.assert_allclose(drr + dtt + dpp, 0.0, atol=1e-14)
    assert np.abs(drt).max() == 0.0


def test_unperturbed_stress_is_G_times_B():
    """Neo-Hookean, inviscid: ``sigma = G B`` with ``B`` known in closed form."""
    field = build_modal_field(_run(ep0=0.0), 60, L, RREF, with_rates=True)
    r, th = _grid(field)
    srr, stt, spp, srt, _, _ = stress_components(field, r, th, Rref=RREF, G=2.0,
                                                 mu=0.0, alpha=0.0)
    rho = np.cbrt(r**3 - field.R**3 + RREF**3)
    np.testing.assert_allclose(srr, 2.0 * rho**4 / r**4, atol=1e-13)
    np.testing.assert_allclose(stt, 2.0 * r**2 / rho**2, atol=1e-13)
    assert np.abs(srt).max() == 0.0


def test_viscous_stress_is_twice_mu_times_the_rate():
    field = build_modal_field(_run(), 60, L, RREF, with_rates=True)
    r, th = _grid(field)
    rate = rate_of_strain_components(field, r, th)
    stress = stress_components(field, r, th, Rref=RREF, G=0.0, mu=0.5, alpha=0.0)
    for s, d in zip(stress, rate):
        np.testing.assert_allclose(s, 2 * 0.5 * d, atol=1e-14)


def test_almansi_reduces_to_small_strain_at_small_amplitude():
    """The two measures differ at first order in the amplitude."""
    deviations = []
    for amp in (1e-3, 1e-2):
        field = build_modal_field(_run(ep0=amp, moving=False), 40, L, RREF)
        r, th = _grid(field)
        a = strain_components(field, r, th, Rref=RREF, measure="almansi")
        s = strain_components(field, r, th, Rref=RREF, measure="small")
        deviations.append(np.abs(a[3] - s[3]).max() / np.abs(s[3]).max())
    # Tenfold smaller amplitude, roughly tenfold closer agreement.
    assert 5.0 < deviations[1] / deviations[0] < 20.0
    assert deviations[0] < 1e-3


def test_modal_coefficients_are_linear_in_the_amplitude():
    base = build_modal_field(_run(modes=[5, 8], ep0=[1e-3, -1e-3]), 60, L, RREF)
    scaled = build_modal_field(_run(modes=[5, 8], ep0=[3e-3, -3e-3]), 60, L, RREF)
    np.testing.assert_allclose(scaled.A, 3 * base.A, rtol=1e-10, atol=1e-18)
    np.testing.assert_allclose(scaled.B, 3 * base.B, rtol=1e-10, atol=1e-18)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_angular_nodal_structure_matches_the_harmonic(n):
    """``e_rr`` follows ``Y_n`` and ``e_rt`` follows ``dY_n/dtheta``.

    So on ``(0, pi)`` the radial component has ``n`` sign changes and the shear
    component ``n - 1``.  This pins the angular reconstruction and the mapping
    of components onto the plot together.
    """
    field = build_modal_field(_run(modes=n, ep0=0.05, moving=False, xN=128), 40, L, RREF)
    th = np.linspace(0.02, np.pi - 0.02, 401)[None, :]
    r = np.full_like(th, field.R * 1.5)

    for scalar, expected in (("err", n), ("ert", n - 1)):
        c = tensor_scalar(strain_components(field, r, th, Rref=RREF), scalar).ravel()
        # The base solution contributes a constant offset to e_rr; remove it.
        if scalar == "err":
            c = c - np.mean(c)
        nz = c[np.abs(c) > 1e-10 * np.abs(c).max()]
        assert np.sum(np.diff(np.sign(nz)) != 0) == expected


def test_incompressibility_of_the_reconstructed_velocity_converges_with_xN():
    """``div v`` is a resolution diagnostic, not an identity.

    The modal velocity is built from the solver's ``T`` field, so its
    divergence inherits the solver's radial resolution.  It converges with
    ``xN``; it does not converge with the plotting grid.  Strain-rate and
    stress figures therefore need a well-resolved run -- this is the test that
    says so.
    """
    th = np.linspace(0.05, np.pi - 0.05, 20)[None, :]
    traces = []
    for xN in (64, 256):
        result = _run(ep0=0.02, xN=xN)
        worst = 0.0
        # Several frames: at any single frame the error can dip by accident, so
        # monotonicity frame-by-frame is not something to assert.
        for frame in (20, 40, 60):
            field = build_modal_field(result, frame, L, RREF, with_rates=True)
            r = np.linspace(field.R * 1.05, 12.0, 400)[:, None]
            d = rate_of_strain_components(field, r, th)
            worst = max(worst, np.abs(d[0] + d[1] + d[2]).max())
        traces.append(worst)

    coarse, fine = traces
    assert fine < coarse / 3.0, f"div v did not converge with xN: {coarse:.2e} -> {fine:.2e}"
    assert fine < 5e-3


# --------------------------------------------------------------------------
# Scalar reduction
# --------------------------------------------------------------------------


def test_tensor_scalar_components_and_invariants():
    comps = (np.array([2.0]), np.array([-1.0]), np.array([0.5]),
             np.array([0.0]), np.array([0.0]), np.array([0.0]))
    assert tensor_scalar(comps, "err") == 2.0
    assert tensor_scalar(comps, "trace") == pytest.approx(1.5)
    assert tensor_scalar(comps, "frobenius") == pytest.approx(np.sqrt(4 + 1 + 0.25))
    assert tensor_scalar(comps, "max_principal") == pytest.approx(2.0)
    assert tensor_scalar(comps, "min_principal") == pytest.approx(-1.0)
    assert tensor_scalar(comps, "signed_maxabs_principal") == pytest.approx(2.0)
    assert tensor_scalar(comps, "maxabs_component") == pytest.approx(2.0)


def test_signed_principal_keeps_the_sign_of_the_largest_magnitude():
    comps = tuple(np.array([v]) for v in (-3.0, 1.0, 0.0, 0.0, 0.0, 0.0))
    assert tensor_scalar(comps, "signed_maxabs_principal") == pytest.approx(-3.0)


def test_principal_values_are_rotation_invariant():
    """Shear in the r-theta plane must not change the principal values."""
    diag = tuple(np.array([v]) for v in (1.0, -1.0, 0.0, 0.0, 0.0, 0.0))
    sheared = tuple(np.array([v]) for v in (0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    assert tensor_scalar(diag, "max_principal") == pytest.approx(1.0)
    assert tensor_scalar(sheared, "max_principal") == pytest.approx(1.0)
    assert tensor_scalar(sheared, "min_principal") == pytest.approx(-1.0)


def test_unknown_scalar_is_rejected():
    comps = tuple(np.zeros(1) for _ in range(6))
    with pytest.raises(ValueError, match="unknown scalar"):
        tensor_scalar(comps, "not_a_scalar")


def test_rates_are_required_before_asking_for_them():
    field = build_modal_field(_run(), 40, L, RREF, with_rates=False)
    r, th = _grid(field)
    with pytest.raises(ValueError, match="with_rates=True"):
        rate_of_strain_components(field, r, th)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def result():
    return _run(modes=[2, 5], ep0=[0.03, -0.02], xN=64, n_steps=60)


def test_field_slice_shape_and_geometry(result):
    from imrns.postprocess.render import FieldSpec, field_slice

    X, Z, C = field_slice(result, 30, L, FieldSpec(), n_radial=40, n_theta=50)
    assert X.shape == Z.shape == C.shape == (40, 50)
    assert np.all(np.isfinite(C))
    # Default outer radius is four wall radii, and nothing is inside the wall.
    assert np.hypot(X, Z).max() == pytest.approx(4 * result.R[30])
    assert np.hypot(X, Z).min() >= result.R[30] * 0.5


@pytest.mark.parametrize("measure,scalar", [
    ("almansi", "ert"), ("small", "err"), ("strain_rate", "ert"), ("stress", "ert"),
    ("almansi", "frobenius"), ("almansi", "signed_maxabs_principal"),
])
def test_every_measure_renders(result, measure, scalar, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from imrns.postprocess.render import FieldSpec, snapshot_grid

    spec = FieldSpec(measure=measure, scalar=scalar, G=2.0, mu=0.015, alpha=0.1)
    out = tmp_path / f"{measure}_{scalar}.png"
    snapshot_grid(result, L, out, spec, n_frames=2, layout=(1, 2),
                  n_radial=25, n_theta=35, dpi=60)
    assert out.exists() and out.stat().st_size > 1000


def test_symmetric_limits_are_centred_on_zero():
    from imrns.postprocess.render import symmetric_limits

    lo, hi = symmetric_limits(np.array([-0.4, 0.1, 0.2, np.nan]))
    assert lo == -hi and hi > 0
    assert symmetric_limits(np.array([np.nan, np.nan]))[1] == 1.0


def test_pick_frames_spans_the_run(result):
    from imrns.postprocess.render import pick_frames

    frames = pick_frames(result, 5)
    assert frames[0] == 0 and frames[-1] == result.t.size - 1
    assert frames == sorted(frames)


def test_field_spec_labels_components():
    from imrns.postprocess.render import FieldSpec

    assert FieldSpec(measure="almansi", scalar="ert").label() == r"$e_{r\theta}$"
    assert FieldSpec(measure="stress", scalar="err").label() == r"$\sigma_{rr}$"
    assert "frobenius" in FieldSpec(scalar="frobenius").label()


def test_colormap_is_luminance_symmetric_about_the_midpoint():
    """Equal magnitudes of either sign must read as equally intense.

    A diverging map whose arms differ in lightness makes one sign look stronger
    than the other at the same magnitude, which is a misreading the viewer
    cannot correct for.
    """
    from imrns.colormaps import brown_centered_diverging, relative_luminance

    Y = relative_luminance(brown_centered_diverging(257))
    np.testing.assert_allclose(Y, Y[::-1], atol=1e-12)
    mid = len(Y) // 2
    assert Y[mid] == min(Y)
    assert np.all(np.diff(Y[mid:]) >= -1e-12), "luminance must rise away from zero"
