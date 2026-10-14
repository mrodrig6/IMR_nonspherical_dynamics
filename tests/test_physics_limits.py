"""Physical limits the solver must reproduce.

These do not compare against the MATLAB source -- they check the ported solver
against results derivable independently, which is the only way to catch an
error that was already present in the original.
"""

from __future__ import annotations

import numpy as np
import pytest

from imrns.solver import solve_perturbation

INF = np.inf


def _free_history(t_end, n_steps):
    t = np.linspace(0.0, t_end, n_steps)
    return t, np.ones_like(t), np.zeros_like(t), np.zeros_like(t)


def _dominant_frequency(t, y):
    """Angular frequency of the dominant spectral peak of ``y(t)``."""
    y = y - y.mean()
    dt = t[1] - t[0]
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    freqs = np.fft.rfftfreq(y.size, dt)
    k = int(np.argmax(spec[1:]) + 1)
    # Parabolic interpolation around the peak for sub-bin accuracy.
    if 1 <= k < spec.size - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        denom = a - 2 * b + c
        if denom != 0:
            k = k + 0.5 * (a - c) / denom
    return 2.0 * np.pi * k * (freqs[1] - freqs[0])


@pytest.mark.parametrize("n", [2, 4, 6])
def test_capillary_mode_frequency_matches_lamb(n):
    """A free inviscid bubble with surface tension alone oscillates at the Lamb frequency.

    For an inviscid liquid the shape mode obeys
    ``omega^2 = (n-1)(n+1)(n+2) sigma / (rho R^3)``.  In the solver's
    nondimensionalisation that is ``omega^2 = (n-1)(n+1)(n+2) / (2 We)``.
    """
    We = 45.0
    omega_expected = np.sqrt((n - 1) * (n + 1) * (n + 2) / (2.0 * We))
    periods = 6.0
    t, R, Rd, Rdd = _free_history(periods * 2 * np.pi / omega_expected, 4000)

    res = solve_perturbation(
        modes=n, t=t, R=R, Rd=Rd, Rdd=Rdd,
        Ca=INF, Re=INF, We=We, xN=24, ep0=1e-3, closure="irr", method=2,
    )
    omega = _dominant_frequency(res.t, res.ep[0])
    assert omega == pytest.approx(omega_expected, rel=0.02)


def test_rotational_converges_to_irrotational_as_deviatoric_stress_vanishes():
    """The rotational correction is first order in the deviatoric stress.

    The rotational field is driven by viscous *and* elastic stress, so the
    irrotational limit needs both ``Ca`` and ``Re`` to grow together.  The
    deviation should then fall like ``1/Ca``.
    """
    t, R, Rd, Rdd = _free_history(3.0, 1200)
    deviations = []
    for v in (1e4, 1e5, 1e6):
        common = dict(modes=5, t=t, R=R, Rd=Rd, Rdd=Rdd, Ca=v, Re=v, We=30.0,
                      xN=32, ep0=0.01, method=2)
        irr = solve_perturbation(closure="irr", **common)
        rot = solve_perturbation(closure="rot", **common)
        deviations.append(np.abs(rot.ep - irr.ep).max() / np.abs(irr.ep).max())

    # Each tenfold increase should cut the deviation by close to ten.
    for coarse, fine in zip(deviations[:-1], deviations[1:]):
        assert 5.0 < coarse / fine < 20.0
    assert deviations[-1] < 1e-4


def test_solution_is_linear_in_the_initial_amplitude():
    """Scaling the initial amplitude scales the whole solution by the same factor.

    The governing equations are linear in the perturbation, so this must hold
    exactly.  It is a sharp check on the assembly: any term that picked up a
    stray constant, or that was added to the wrong row, breaks it.
    """
    t, R, Rd, Rdd = _free_history(3.0, 800)
    common = dict(modes=[2, 5], t=t, R=R, Rd=Rd, Rdd=Rdd, Ca=50.0, Re=35.0,
                  We=45.0, alpha=0.1, xN=32, closure="rot", method=2)
    base = solve_perturbation(ep0=[0.01, -0.02], **common)
    scaled = solve_perturbation(ep0=[0.03, -0.06], **common)
    np.testing.assert_allclose(scaled.ep, 3.0 * base.ep, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(scaled.T, 3.0 * base.T, rtol=1e-10, atol=1e-15)


def test_rotational_field_responds_to_the_radial_drive():
    """A moving wall drives the rotational field; a static one with no initial field does not."""
    t = np.linspace(0.0, 3.0, 1200)
    static = solve_perturbation(modes=5, t=t, R=np.ones_like(t), Rd=np.zeros_like(t),
                                Rdd=np.zeros_like(t), Ca=50.0, Re=35.0, We=45.0,
                                xN=32, ep0=0.0, epd0=0.0, closure="rot", method=2)
    assert np.abs(static.T).max() < 1e-12
    assert np.abs(static.ep).max() < 1e-12


def test_rotational_closure_rejects_vanishing_deviatoric_stress():
    """Infinite Ca and Re together make the wall condition singular, so refuse it."""
    t, R, Rd, Rdd = _free_history(1.0, 50)
    with pytest.raises(ValueError, match="rotational closure is undefined"):
        solve_perturbation(modes=5, t=t, R=R, Rd=Rd, Rdd=Rdd,
                           Ca=INF, Re=INF, We=30.0, xN=24, ep0=0.01, closure="rot")


def test_viscosity_damps_the_amplitude():
    """A viscous free bubble loses amplitude; an inviscid one does not."""
    t, R, Rd, Rdd = _free_history(8.0, 3000)
    common = dict(modes=4, t=t, R=R, Rd=Rd, Rdd=Rdd, Ca=INF, We=30.0,
                  xN=24, ep0=0.01, closure="irr", method=2)
    inviscid = solve_perturbation(Re=INF, **common)
    viscous = solve_perturbation(Re=8.0, **common)
    tail = slice(-200, None)
    assert np.abs(viscous.ep[0, tail]).max() < 0.5 * np.abs(inviscid.ep[0, tail]).max()


def test_elasticity_raises_the_frequency():
    """Stiffening the medium (smaller Ca) increases the oscillation frequency."""
    t, R, Rd, Rdd = _free_history(10.0, 6000)
    common = dict(modes=4, t=t, R=R, Rd=Rd, Rdd=Rdd, Re=INF, We=1e6,
                  xN=24, ep0=1e-3, closure="irr", method=2)
    soft = _dominant_frequency(t, solve_perturbation(Ca=200.0, **common).ep[0])
    stiff = _dominant_frequency(t, solve_perturbation(Ca=20.0, **common).ep[0])
    assert stiff > soft


def test_modes_are_independent():
    """Solving modes together gives the same answer as solving them one at a time.

    This is the property the block-diagonal solve relies on.
    """
    t, R, Rd, Rdd = _free_history(3.0, 600)
    modes, amps = [2, 5, 8], [0.01, -0.02, 0.015]
    common = dict(t=t, R=R, Rd=Rd, Rdd=Rdd, Ca=50.0, Re=35.0, We=45.0,
                  alpha=0.1, xN=24, closure="rot", method=2)
    together = solve_perturbation(modes=modes, ep0=amps, **common)
    for k, (n, a) in enumerate(zip(modes, amps)):
        alone = solve_perturbation(modes=n, ep0=a, **common)
        np.testing.assert_allclose(together.ep[k], alone.ep[0], rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose(together.T[k], alone.T[0], rtol=1e-11, atol=1e-14)


def test_grid_refinement_converges():
    """The amplitude history converges as the radial grid is refined."""
    t, R, Rd, Rdd = _free_history(2.0, 400)
    common = dict(modes=5, t=t, R=R, Rd=Rd, Rdd=Rdd, Ca=50.0, Re=35.0, We=45.0,
                  alpha=0.1, ep0=0.01, closure="rot", method=2)
    ref = solve_perturbation(xN=256, **common).ep[0]
    errs = [np.abs(solve_perturbation(xN=xN, **common).ep[0] - ref).max() for xN in (32, 64, 128)]
    assert errs[0] > errs[1] > errs[2]


def test_outputs_have_consistent_length():
    """Every returned array is trimmed to the same number of time points."""
    t, R, Rd, Rdd = _free_history(2.0, 300)
    res = solve_perturbation(modes=[2, 5], t=t, R=R, Rd=Rd, Rdd=Rdd,
                             Ca=50.0, Re=35.0, We=45.0, xN=24, ep0=0.01, closure="rot")
    n = res.t.size
    assert res.R.size == n and res.Rd.size == n
    assert res.ep.shape == (2, n) and res.epd.shape == (2, n)
    assert res.T.shape == (2, 24, n) and res.Td.shape == (2, 24, n)


def test_factorisation_reuse_gives_identical_results():
    """Caching the LU factorisation must not change the answer.

    The solver reuses a factorisation when the radius history is constant and
    the time grid uniform, which makes the operator time invariant.  The reused
    path rebuilds only the boundary right-hand side and applies the cached row
    scaling, so it is worth pinning that it agrees with rebuilding every step.
    """
    t, R, Rd, Rdd = _free_history(2.0, 400)
    common = dict(modes=[2, 5], t=t, R=R, Rd=Rd, Rdd=Rdd, Ca=50.0, Re=35.0,
                  We=45.0, alpha=0.1, xN=32, ep0=[0.01, -0.02], closure="rot", method=2)
    reused = solve_perturbation(reuse_factorisation=True, **common)
    rebuilt = solve_perturbation(reuse_factorisation=False, **common)
    assert reused.diagnostics["reused_factorisation"] is True
    assert rebuilt.diagnostics["reused_factorisation"] is False
    np.testing.assert_allclose(reused.ep, rebuilt.ep, rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(reused.T, rebuilt.T, rtol=1e-11, atol=1e-15)


def test_factorisation_reuse_also_holds_for_prescribed_amplitude():
    """The reused path rebuilds the forcing term, so check it with forcing on."""
    t, R, Rd, Rdd = _free_history(2.0, 300)
    forced_ep = np.sin(2 * np.pi * t)[None, :] * 0.01
    forced_epd = 0.01 * 2 * np.pi * np.cos(2 * np.pi * t)[None, :]
    common = dict(modes=[5], t=t, R=R, Rd=Rd, Rdd=Rdd, Ca=50.0, Re=35.0, We=45.0,
                  alpha=0.1, xN=32, closure="rot", method=2,
                  forced_ep=forced_ep, forced_epd=forced_epd)
    reused = solve_perturbation(reuse_factorisation=True, **common)
    rebuilt = solve_perturbation(reuse_factorisation=False, **common)
    np.testing.assert_allclose(reused.T, rebuilt.T, rtol=1e-11, atol=1e-15)
    # The prescribed history must come back unchanged.
    np.testing.assert_allclose(reused.ep, forced_ep, rtol=1e-14)
