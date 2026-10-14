"""Check every coefficient kernel against the MATLAB text it was ported from.

Each test evaluates the expression extracted from ``matlab_legacy/common/*.m``
and the corresponding function in :mod:`imrns.assemble.kernels` on the same
inputs, and requires agreement to near machine precision.  See
:mod:`matlab_oracle` for why the comparison is made against the source text
rather than against transcribed constants.
"""

from __future__ import annotations

import numpy as np
import pytest
from matlab_oracle import Indexable, evaluate
from scipy.special import hyp2f1

from imrns.assemble import kernels as K

# A state deliberately away from any special value: expanded bubble, moving
# wall, finite acceleration, both viscous and elastic stresses active.
R, Rd, Rdd, Req = 1.37, 0.41, -0.23, 1.0
Ca, Re, We, alph = 7.3, 23.0, 11.5, 0.17
INV_CA, INV_RE, INV_WE = 1.0 / Ca, 1.0 / Re, 1.0 / We
MODES = [2.0, 5.0, 8.0, 11.0]

r = np.linspace(R, 40.0, 48)
rs = np.cbrt(r**3 - R**3 + Req**3)
vs = Req**3 * ((R / Req) ** 3 - 1.0)
sr = R / Req


def base_env(n):
    return dict(
        r=r, rs=rs, vs=vs, R=R, Rd=Rd, Rdd=Rdd, Req=Req, Ro=Req,
        n=n, Ca=Ca, Re=Re, We=We, alph=alph, alpha=alph, sr=sr,
        U=Rd, Rddot=Rdd, lam=sr, s=r, ds=None,
    )


def assert_close(got, want, label):
    got, want = np.asarray(got, float), np.asarray(want, float)
    got = np.broadcast_to(got, want.shape) if got.shape != want.shape else got
    assert np.all(np.isfinite(want)), f"{label}: MATLAB reference not finite"
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-13, err_msg=label)


@pytest.mark.parametrize("n", MODES)
def test_non_int_coeffs(n):
    got = K.non_int_coeffs(r, R, Rd, Rdd, n, INV_CA, INV_RE, alph, rs, vs)
    env = base_env(n)
    for i, name in enumerate(["Ct1", "Ct2", "Ct3", "Ct4", "Ct5", "Ct6", "Ct7"]):
        assert_close(got[i], evaluate("f_non_int_coeffs.m", name, env), f"n={n} {name}")


@pytest.mark.parametrize("n", MODES)
def test_kappa_coeffs(n):
    Ckap, Ckapdot = K.kappa_coeffs(r, R, Rd, Rdd, n, INV_CA, INV_RE, alph, rs, vs)
    env = base_env(n)
    assert_close(Ckap, evaluate("f_bulk_kappa.m", "Ckap", env), f"n={n} Ckap")
    assert_close(Ckapdot, evaluate("f_bulk_kappa.m", "Ckapdot", env), f"n={n} Ckapdot")


@pytest.mark.parametrize("n", MODES)
def test_KJ_coeffs(n):
    CJ, CK, CJd, CKd = K.KJ_coeffs(r, rs, vs, n, R, Rd, Rdd, INV_CA, INV_RE, alph)
    env = base_env(n)
    for got, name in [(CJ, "CJ"), (CK, "CK"), (CJd, "CJd"), (CKd, "CKd")]:
        assert_close(got, evaluate("f_bulk_KJ.m", name, env), f"n={n} {name}")


@pytest.mark.parametrize("n", MODES)
def test_Teq_epsilon_coeffs(n):
    ep_coef, epd_coef = K.Teq_epsilon_coeffs(R, Rd, Rdd, r, n, INV_CA, alph, INV_RE, rs, vs)
    # The MATLAB function transposes r and rs on entry; the values are the same.
    env = base_env(n)
    assert_close(ep_coef, evaluate("f_Teq_epsilon_coefs.m", "epCoef", env), f"n={n} epCoef")
    assert_close(epd_coef, evaluate("f_Teq_epsilon_coefs.m", "epdCoef", env), f"n={n} epdCoef")


@pytest.mark.parametrize("n", MODES)
def test_forced_source(n):
    ep, epd = 0.031, -0.014
    got = K.forced_source(ep, epd, R, Rd, Rdd, r, n, INV_CA, INV_RE, alph, rs, vs)
    env = base_env(n)
    # The MATLAB source indexes the per-mode arrays as ep(i), epd(i).
    env.update(ep=Indexable(ep), epd=Indexable(epd), i=1)
    src = evaluate("f_source_terms.m", "Sp", env) - evaluate("f_source_terms.m", "Ap", env)
    assert_close(got, src, f"n={n} Sp-Ap")


def _hyp_values(n, stretch):
    return [hyp2f1(a, b, c, z) for a, b, c, z in K.hyp_arguments(n, stretch)]


@pytest.mark.parametrize("n", MODES)
def test_epsilon_coeffs_me(n):
    F = _hyp_values(n, sr)
    xi, eta = K.epsilon_coeffs_me(R, Rd, sr, Rdd, n, Req, INV_WE, INV_RE, INV_CA, alph, F)
    env = base_env(n)
    elast = evaluate("f_epsilon_coeffs.m", "elast", env)
    sselast = evaluate("f_epsilon_coeffs.m", "sselast", env)
    visceta = evaluate("f_epsilon_coeffs.m", "visceta", env)
    viscxi = evaluate("f_epsilon_coeffs.m", "viscxi", env)
    surften = evaluate("f_epsilon_coeffs.m", "surften", env)
    eta_ml = 4 * (5 + 2 * n) * Rd / ((4 + n) * R) + visceta / Re
    xi_ml = (
        6 * (n + 2) / (n + 4) * Rd**2 / R**2
        + (8 + n - n**2) * Rdd / ((n + 4) * R)
        + elast / Ca
        + alph / Ca * sselast
        + surften / (2 * We)
        + viscxi / Re
    )
    assert_close(xi, xi_ml, f"n={n} xi")
    assert_close(eta, eta_ml, f"n={n} eta")


@pytest.mark.parametrize("n", MODES)
def test_epsilon_coeffs_pros(n):
    xi, eta = K.epsilon_coeffs_pros(R, Rd, sr, Rdd, n, INV_WE, INV_RE, INV_CA)
    env = base_env(n)
    adot = evaluate("f_epsilon_coeffs.m", "adot_prefact", env)
    env["adot_prefact"] = adot
    apre = evaluate("f_epsilon_coeffs.m", "aprefact", env)
    assert_close(xi, apre + adot * Rd / R + Rdd / R, f"n={n} pros xi")
    assert_close(eta, adot + 2 * Rd / R, f"n={n} pros eta")


@pytest.mark.parametrize("n", MODES)
def test_ep_feedback_pointwise(n):
    fT, fV, fdT = K.ep_feedback_pointwise(r, R, Rd, n, INV_CA, INV_RE, alph, rs)
    env = base_env(n)
    assert_close(fT, evaluate("f_ep_source_terms.m", "fTPI", env), f"n={n} fTPI")
    assert_close(fV, evaluate("f_ep_source_terms.m", "fVPI", env), f"n={n} fVPI")
    assert_close(fdT, evaluate("f_ep_source_terms.m", "fdTdrPI", env), f"n={n} fdTdrPI")


@pytest.mark.parametrize("n", MODES)
def test_ep_feedback_kappa(n):
    F = _hyp_values(n, sr)
    H, G, Fk, fTw = K.ep_feedback_kappa(R, Rd, Rdd, sr, n, INV_CA, INV_RE, alph, F)
    env = base_env(n)
    assert_close(H, evaluate("f_ep_source_terms.m", "Hkapddot", env), f"n={n} Hkapddot")
    assert_close(G, evaluate("f_ep_source_terms.m", "Gkapdot", env), f"n={n} Gkapdot")
    assert_close(Fk, evaluate("f_ep_source_terms.m", "Fkap", env), f"n={n} Fkap")
    assert_close(fTw, evaluate("f_ep_source_terms.m", "fT", env), f"n={n} fT")


@pytest.mark.parametrize("n", MODES)
def test_ep_feedback_J_K(n):
    fJT, fJdT, fJdt2 = K.ep_feedback_J(r, R, Rd, Rdd, Req, sr, n, INV_CA, INV_RE, alph, rs, vs)
    fKT, fKdT, fKdt2 = K.ep_feedback_K(r, R, Rd, Rdd, sr, n, INV_CA, INV_RE, alph, rs, vs)
    env = base_env(n)
    assert_close(fJT, evaluate("f_ep_source_terms.m", "fJT", env), f"n={n} fJT")
    assert_close(fJdT, evaluate("f_ep_source_terms.m", "fJdT", env), f"n={n} fJdT")
    assert_close(fJdt2, evaluate("f_ep_source_terms.m", "fJdt2", env), f"n={n} fJdt2")
    assert_close(fKT, evaluate("f_ep_source_terms.m", "fKT", env), f"n={n} fKT")
    assert_close(fKdT, evaluate("f_ep_source_terms.m", "fKdT", env), f"n={n} fKdT")
    assert_close(fKdt2, evaluate("f_ep_source_terms.m", "fKdt2", env), f"n={n} fKdt2")


@pytest.mark.parametrize("n", MODES)
def test_wall_terms(n):
    zeta, fT = K.wall_traction(sr, Req, R, Rd, INV_CA, INV_RE, alph)
    env = base_env(n)
    zeta_ml = evaluate("f_boundary_conditions.m", "zeta", env)
    env["zeta"] = zeta_ml
    assert_close(zeta, zeta_ml, f"n={n} zeta")
    assert_close(fT, evaluate("f_boundary_conditions.m", "fT", env), f"n={n} wall fT")

    fk, j2k, j2kd = K.wall_kappa_terms(n, R, Rd, sr, zeta, INV_RE)
    assert_close(fk, evaluate("f_boundary_conditions.m", "fkappadopt", env), f"n={n} fkappadopt")
    assert_close(j2k, evaluate("f_boundary_conditions.m", "j2kappa", env), f"n={n} j2kappa")
    assert_close(j2kd, evaluate("f_boundary_conditions.m", "j2kappadot", env), f"n={n} j2kappadot")


def test_wall_traction_reduces_at_unit_stretch():
    """At sr = 1 the elastic traction is Req/Ca regardless of alph."""
    for a in (0.0, 0.3, 1.2):
        zeta, _ = K.wall_traction(1.0, Req, R, Rd, INV_CA, INV_RE, a)
        assert zeta == pytest.approx(Req / Ca, rel=1e-14)
