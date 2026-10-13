"""Reconstruct displacement, strain, strain-rate and stress fields.

This is the physics half of the visualisation; :mod:`imrns.postprocess.render`
draws the results and imports nothing from matplotlib here.  The MATLAB original
(``matlab_legacy/common/make_axisym_displacement_movie_all_fields.m``, 2769
lines) interleaved the two, which made the field reconstruction impossible to
test on its own.  Everything in this module returns arrays.

What is reconstructed
---------------------
The solver returns the modal amplitudes ``eps_n(t)`` and the rotational field
``T_n(r, t)``.  The displacement in the medium is

    u_r     = u_base(r) + sum_n A_n(r) Y_n(theta, phi)
    u_theta =             sum_n B_n(r) dY_n/dtheta
    u_phi   =             sum_n B_n(r) dY_n/dphi / sin(theta)

with a spherically symmetric base part fixed by incompressibility,

    u_base = r - rho,    rho = (r^3 - R^3 + Rref^3)^(1/3),

and the modal coefficients built from ``T_n`` through the same ``J``, ``K`` and
``kappa`` radial kernels the solver uses:

    Phi_n  = (n+1)/(2n+1) K r^n + [ n/(n+1) R^(2n+1) kappa + n/(2n+1) J ] r^-(n+1)
    u_rot  = n R^(2n+1) kappa r^-(n+2) + n(n+1)/(2n+1) [ J r^-(n+2) - K r^(n-1) ]
    P_n    = eps_n R^(n+3)
    A_n    = P_n r^-(n+2) + u_rot
    B_n    = -[ P_n / ((n+1) r^(n+2)) + Phi_n / r ]

The first two terms of ``A_n`` are the potential-flow and rotational parts of
the radial displacement respectively; ``B_n`` is the tangential coefficient.

Measures available
------------------
``small``
    Symmetric part of the spatial displacement gradient.
``almansi``
    Eulerian Almansi strain ``e = (I - B^-1)/2`` with ``B^-1 = F^-T F^-1`` and
    ``F^-1 = I - h``.  This is the default and the one the paper figures use.
``strain_rate``
    Rate of deformation ``D = (grad v + grad v^T)/2``.
``stress``
    ``sigma = G (1 + alpha (I_B - 3)) B + 2 mu D - P I``, matching the
    constitutive model the solver is derived for.  Supplying ``pressure`` gives
    the full Cauchy stress; omitting it gives the constitutive part, which is
    the same thing for shear components.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import eval_legendre, lpmv

from ..grid import computational_grid, drdx as _drdx, radius as _radius
from ..operators import trapezoid_matrices

__all__ = [
    "ModalField",
    "build_modal_field",
    "harmonics",
    "displacement_gradient",
    "strain_components",
    "rate_of_strain_components",
    "stress_components",
    "tensor_scalar",
    "SCALAR_ALIASES",
]

# Below this, sin(theta) is clamped so the polar axis does not divide by zero.
POLE_EPS = 1e-6


# --------------------------------------------------------------------------
# Spherical harmonics
# --------------------------------------------------------------------------


def _normalisation(n: int, m: int) -> float:
    """Real orthonormal spherical-harmonic normalisation constant."""
    from scipy.special import gammaln

    if m == 0:
        return float(np.sqrt((2 * n + 1) / (4 * np.pi)))
    return float(
        np.sqrt(2.0)
        * np.sqrt((2 * n + 1) / (4 * np.pi) * np.exp(gammaln(n - m + 1) - gammaln(n + m + 1)))
    )


def harmonics(n: int, theta, az=None, mode: str = "axisymmetric", part: str = "cos"):
    """Return ``(Y, Y_t, Y_tt, Y_p, Y_tp, Y_pp)`` for mode ``n``.

    Subscripts are derivatives with respect to ``theta`` and ``phi``.

    ``mode`` selects the angular reconstruction: ``"axisymmetric"`` uses the
    ``m = 0`` harmonic, which is what the solver's linearisation assumes;
    ``"sectoral"`` uses the real ``m = n`` harmonic, available for
    cross-sections.  ``part`` picks ``cos(m phi)`` or ``sin(m phi)``.

    The second theta-derivative is obtained from the spherical-harmonic
    identity ``Y_tt + cot(theta) Y_t + Y_pp / sin^2(theta) + n(n+1) Y = 0``
    rather than by differentiating twice, which is both cheaper and better
    behaved near the poles.
    """
    theta = np.asarray(theta, dtype=float)
    x = np.cos(theta)
    s = np.sin(theta)
    s_safe = np.maximum(np.abs(s), POLE_EPS)
    cot = np.cos(theta) / s_safe

    mode = mode.lower()
    if mode in ("axisymmetric", "axisym", "m0"):
        m = 0
        c = _normalisation(n, 0)
        P = eval_legendre(n, x)
        Y = c * P
        if n == 0:
            Yt = np.zeros_like(Y)
        else:
            Yt = c * n * (x * P - eval_legendre(n - 1, x)) / s_safe
        Yp = np.zeros_like(Y)
        Ytp = np.zeros_like(Y)
        Ypp = np.zeros_like(Y)
    elif mode in ("sectoral", "m_equals_n", "m=n"):
        if az is None:
            raise ValueError("the sectoral reconstruction needs an azimuth")
        m = n
        az = np.asarray(az, dtype=float)
        c = _normalisation(n, m)
        Pnm = lpmv(m, n, x)
        Pn1m = lpmv(m, n - 1, x) if n - 1 >= m else np.zeros_like(Pnm)
        dP = (n * x * Pnm - (n + m) * Pn1m) / s_safe
        if part.lower() in ("cos", "cosine"):
            trig, trig_p, trig_pp = np.cos(m * az), -m * np.sin(m * az), -(m**2) * np.cos(m * az)
        elif part.lower() in ("sin", "sine"):
            trig, trig_p, trig_pp = np.sin(m * az), m * np.cos(m * az), -(m**2) * np.sin(m * az)
        else:
            raise ValueError("part must be 'cos' or 'sin'")
        Y = c * Pnm * trig
        Yt = c * dP * trig
        Yp = c * Pnm * trig_p
        Ytp = c * dP * trig_p
        Ypp = c * Pnm * trig_pp
    else:
        raise ValueError(f"unknown angular mode {mode!r}; use 'axisymmetric' or 'sectoral'")

    Ytt = -cot * Yt - Ypp / s_safe**2 - n * (n + 1) * Y

    out = [Y, Yt, Ytt, Yp, Ytp, Ypp]
    return tuple(np.where(np.isfinite(a), a, 0.0) for a in out)


# --------------------------------------------------------------------------
# Modal displacement coefficients
# --------------------------------------------------------------------------


@dataclass
class ModalField:
    """Modal displacement coefficients on the solver's radial grid at one frame.

    ``A`` and ``B`` are the radial and tangential coefficients, ``Ar``/``Br``
    their radial derivatives, and ``At``/``Bt`` their time derivatives at fixed
    current ``r`` (needed only for strain-rate and stress).
    """

    r: np.ndarray
    A: np.ndarray
    B: np.ndarray
    Ar: np.ndarray
    Br: np.ndarray
    At: np.ndarray | None
    Bt: np.ndarray | None
    modes: tuple[int, ...]
    R: float
    Rdot: float
    frame: int


def _kernels(r, drdx, n, W, w, One_wT):
    """The ``K``, ``J`` and ``kappa`` quadrature operators, as in the solver."""
    KmT = (One_wT - W) * (-(r ** (-n)) * drdx)[None, :]
    JmT = W * (r ** (n + 1.0) * drdx)[None, :]
    wkap = w * (-(n + 1.0) / (2.0 * n + 1.0) * r ** (-n) * drdx)
    return KmT, JmT, wkap


def _coefficients(n, K, J, kappa, r, R, ep_n):
    """Assemble ``(A, B)`` from the kernel integrals. See the module docstring."""
    Phi = ((n + 1.0) / (2.0 * n + 1.0)) * K * r**n + (
        (n / (n + 1.0)) * R ** (2 * n + 1) * kappa + (n / (2.0 * n + 1.0)) * J
    ) * r ** (-(n + 1.0))

    u_rot = n * R ** (2 * n + 1) * kappa * r ** (-(n + 2.0)) + (
        n * (n + 1.0) / (2.0 * n + 1.0)
    ) * (J * r ** (-(n + 2.0)) - K * r ** (n - 1.0))
    u_rot = u_rot.copy()
    u_rot[0] = 0.0  # T - dPhi/dr vanishes at the mean wall

    P_n = ep_n * R ** (n + 3.0)
    A = P_n * r ** (-(n + 2.0)) + u_rot
    B = -(P_n / ((n + 1.0) * r ** (n + 2.0)) + Phi / r)
    return A, B


def build_modal_field(result, frame: int, L: float, Rref: float = 1.0,
                      with_rates: bool = False) -> ModalField:
    """Build the modal coefficients at one output frame.

    ``result`` is a :class:`imrns.solver.PerturbationResult`.  Set
    ``with_rates`` for strain-rate or stress, which need the time derivatives;
    they are computed analytically from the solver's own ``Td`` and ``Rd``
    rather than by differencing saved frames.
    """
    T, ep, R, t = result.T, result.ep, result.R, result.t
    modes = result.modes
    xN = T.shape[1]
    it = int(frame)

    x = computational_grid(xN)
    r = _radius(x, R[it], L)
    drdx = _drdx(x, R[it], L)
    W, w, One_wT = trapezoid_matrices(x)

    n_modes = len(modes)
    A = np.zeros((n_modes, xN))
    B = np.zeros((n_modes, xN))
    Ar = np.zeros((n_modes, xN))
    Br = np.zeros((n_modes, xN))

    for j, n in enumerate(modes):
        n = float(n)
        KmT, JmT, wkap = _kernels(r, drdx, n, W, w, One_wT)
        Tcol = T[j, :, it]
        A[j], B[j] = _coefficients(n, KmT @ Tcol, JmT @ Tcol, wkap @ Tcol, r, R[it], ep[j, it])
        Ar[j] = np.gradient(A[j], r)
        Br[j] = np.gradient(B[j], r)

    Rdot = float(result.Rd[it])
    At = Bt = None
    if with_rates:
        At, Bt = _coefficient_rates(result, it, r, drdx, L, W, w, One_wT, Rdot)

    return ModalField(r=r, A=A, B=B, Ar=Ar, Br=Br, At=At, Bt=Bt,
                      modes=tuple(modes), R=float(R[it]), Rdot=Rdot, frame=it)


def _coefficient_rates(result, it, r, drdx, L, W, w, One_wT, Rdot):
    """Time derivatives of ``A`` and ``B`` at fixed current ``r``.

    Differentiating the kernel integrals is not just a matter of applying them
    to ``Td``: ``J`` and ``kappa`` have the moving wall ``R(t)`` as their lower
    limit, so each picks up a Leibniz term proportional to ``Rdot T(R, t)``.
    ``K`` integrates outward from ``r``, whose lower limit is fixed under
    differentiation at fixed current ``r``, so it does not.
    """
    T, Td, ep, epd, R = result.T, result.Td, result.ep, result.epd, result.R
    modes = result.modes
    xN = T.shape[1]
    At = np.zeros((len(modes), xN))
    Bt = np.zeros((len(modes), xN))

    for j, n in enumerate(modes):
        n = float(n)
        KmT, JmT, wkap = _kernels(r, drdx, n, W, w, One_wT)
        Tcol, Tdcol = T[j, :, it], Td[j, :, it]

        kappa = wkap @ Tcol
        Kt, Jt, kappat = KmT @ Tdcol, JmT @ Tdcol, wkap @ Tdcol

        T_wall = Tcol[0]
        Jt = Jt - Rdot * R[it] ** (n + 1.0) * T_wall
        kappat = kappat + ((n + 1.0) / (2.0 * n + 1.0)) * Rdot * R[it] ** (-n) * T_wall

        common = (2 * n + 1.0) * R[it] ** (2 * n) * Rdot * kappa + R[it] ** (2 * n + 1) * kappat

        Phit = ((n + 1.0) / (2.0 * n + 1.0)) * Kt * r**n + (
            (n / (n + 1.0)) * common + (n / (2.0 * n + 1.0)) * Jt
        ) * r ** (-(n + 1.0))

        u_rot_t = n * common * r ** (-(n + 2.0)) + (n * (n + 1.0) / (2.0 * n + 1.0)) * (
            Jt * r ** (-(n + 2.0)) - Kt * r ** (n - 1.0)
        )
        u_rot_t = u_rot_t.copy()
        u_rot_t[0] = 0.0

        P_nt = epd[j, it] * R[it] ** (n + 3.0) + ep[j, it] * (n + 3.0) * R[it] ** (n + 2.0) * Rdot

        At[j] = P_nt * r ** (-(n + 2.0)) + u_rot_t
        Bt[j] = -(P_nt / ((n + 1.0) * r ** (n + 2.0)) + Phit / r)

    return At, Bt


# --------------------------------------------------------------------------
# Field evaluation on a plotting grid
# --------------------------------------------------------------------------


def _interp(field: ModalField, values, r_query):
    """Interpolate a modal coefficient onto the plotting radii."""
    return np.interp(r_query, field.r, values, left=values[0], right=values[-1])


def displacement_gradient(field: ModalField, r_grid, theta, az=None, Rref: float = 1.0,
                          mode: str = "axisymmetric", part: str = "cos"):
    """Spatial displacement gradient ``h = du/dx`` in the spherical basis.

    Returns the nine components in the order
    ``(hrr, hrt, hrp, htr, htt, htp, hpr, hpt, hpp)``, each shaped like
    ``r_grid``.  ``r_grid`` and ``theta`` broadcast against each other.
    """
    r = np.maximum(np.real(np.asarray(r_grid, dtype=float)), np.finfo(float).eps)
    theta = np.asarray(theta, dtype=float)
    s = np.sin(theta)
    s_safe = np.maximum(np.abs(s), POLE_EPS)
    cot = np.cos(theta) / s_safe

    # Spherically symmetric base displacement, fixed by incompressibility.
    rho = np.cbrt(np.maximum(r**3 - field.R**3 + Rref**3, np.finfo(float).eps))
    ur = r - rho
    dur_dr = 1.0 - r**2 / np.maximum(rho**2, np.finfo(float).eps)

    zeros = np.zeros_like(r * theta)
    uth = zeros.copy()
    uph = zeros.copy()
    dur_dth = zeros.copy()
    dur_dph = zeros.copy()
    duth_dr = zeros.copy()
    duth_dth = zeros.copy()
    duth_dph = zeros.copy()
    duph_dr = zeros.copy()
    duph_dth = zeros.copy()
    duph_dph = zeros.copy()
    ur = ur + zeros
    dur_dr = dur_dr + zeros

    for j, n in enumerate(field.modes):
        Y, Yt, Ytt, Yp, Ytp, Ypp = harmonics(int(n), theta, az, mode=mode, part=part)
        A = _interp(field, field.A[j], r)
        Ar = _interp(field, field.Ar[j], r)
        B = _interp(field, field.B[j], r)
        Br = _interp(field, field.Br[j], r)

        ur = ur + A * Y
        dur_dr = dur_dr + Ar * Y
        dur_dth = dur_dth + A * Yt
        dur_dph = dur_dph + A * Yp

        uth = uth + B * Yt
        duth_dr = duth_dr + Br * Yt
        duth_dth = duth_dth + B * Ytt
        duth_dph = duth_dph + B * Ytp

        uph = uph + B * (Yp / s_safe)
        duph_dr = duph_dr + Br * (Yp / s_safe)
        duph_dth = duph_dth + B * (Ytp / s_safe - Yp * np.cos(theta) / s_safe**2)
        duph_dph = duph_dph + B * (Ypp / s_safe)

    hrr = dur_dr
    hrt = dur_dth / r - uth / r
    hrp = dur_dph / (r * s_safe) - uph / r
    htr = duth_dr
    htt = ur / r + duth_dth / r
    htp = duth_dph / (r * s_safe) - uph * cot / r
    hpr = duph_dr
    hpt = duph_dth / r
    hpp = duph_dph / (r * s_safe) + ur / r + uth * cot / r
    return hrr, hrt, hrp, htr, htt, htp, hpr, hpt, hpp


def strain_components(field: ModalField, r_grid, theta, az=None, Rref: float = 1.0,
                      measure: str = "almansi", mode: str = "axisymmetric", part: str = "cos"):
    """Strain components ``(err, ett, epp, ert, erp, etp)``.

    ``measure`` is ``"almansi"`` for the Eulerian Almansi strain
    ``e = (I - B^-1)/2``, or ``"small"`` for the symmetric displacement
    gradient.
    """
    h = displacement_gradient(field, r_grid, theta, az, Rref, mode, part)
    hrr, hrt, hrp, htr, htt, htp, hpr, hpt, hpp = h

    measure = measure.lower()
    if measure in ("small", "linear", "infinitesimal", "engineering"):
        return (hrr, htt, hpp, 0.5 * (hrt + htr), 0.5 * (hrp + hpr), 0.5 * (htp + hpt))

    if measure not in ("almansi", "eulerian_almansi", "eulerian", "finite"):
        raise ValueError(f"unknown strain measure {measure!r}; use 'almansi' or 'small'")

    # F^-1 = I - h, and B^-1 = F^-T F^-1, so B^-1 is the Gram matrix of the
    # columns of (I - h).
    m11, m12, m13 = 1.0 - hrr, -hrt, -hrp
    m21, m22, m23 = -htr, 1.0 - htt, -htp
    m31, m32, m33 = -hpr, -hpt, 1.0 - hpp

    b11 = m11**2 + m21**2 + m31**2
    b12 = m11 * m12 + m21 * m22 + m31 * m32
    b13 = m11 * m13 + m21 * m23 + m31 * m33
    b22 = m12**2 + m22**2 + m32**2
    b23 = m12 * m13 + m22 * m23 + m32 * m33
    b33 = m13**2 + m23**2 + m33**2

    return (0.5 * (1 - b11), 0.5 * (1 - b22), 0.5 * (1 - b33),
            -0.5 * b12, -0.5 * b13, -0.5 * b23)


def rate_of_strain_components(field: ModalField, r_grid, theta, az=None,
                              mode: str = "axisymmetric", part: str = "cos"):
    """Rate of deformation ``D = (grad v + grad v^T)/2``.

    The velocity has a spherically symmetric base part ``R^2 Rdot / r^2`` from
    the radial motion, plus modal contributions.  Its radial derivative is
    taken analytically, so an unperturbed bubble gives the exact incompressible
    answer rather than the plotting grid's truncation error.  The modal velocity
    coefficients are *not* simply ``At`` and ``Bt``: holding the current radius
    fixed while the base flow carries material through it adds advective terms,

        U_r = At + v_s Ar + (2 R^2 Rdot / r^3) A
        U_t = Bt + v_s Br - v_s B / r

    with ``v_s = R^2 Rdot / r^2``.
    """
    if field.At is None:
        raise ValueError(
            "strain rate needs the coefficient time derivatives; "
            "rebuild the field with build_modal_field(..., with_rates=True)"
        )
    r = np.maximum(np.real(np.asarray(r_grid, dtype=float)), np.finfo(float).eps)
    theta = np.asarray(theta, dtype=float)
    s_safe = np.maximum(np.abs(np.sin(theta)), POLE_EPS)
    cot = np.cos(theta) / s_safe

    base_flux = field.R**2 * field.Rdot
    v_s = base_flux / r**2

    zeros = np.zeros_like(r * theta)
    # The base radial velocity is known in closed form, so its radial
    # derivative is taken analytically and only the modal part is differenced.
    # The MATLAB differenced the whole thing, which left D_rr carrying the
    # plotting grid's truncation error even for a purely spherical bubble.
    vr_modal = zeros.copy()
    vr = v_s + zeros
    vth = zeros.copy()
    vph = zeros.copy()
    dvr_dth = zeros.copy()
    dvr_dph = zeros.copy()
    dvth_dth = zeros.copy()
    dvth_dph = zeros.copy()
    dvph_dth = zeros.copy()
    dvph_dph = zeros.copy()

    for j, n in enumerate(field.modes):
        Y, Yt, Ytt, Yp, Ytp, Ypp = harmonics(int(n), theta, az, mode=mode, part=part)
        A = _interp(field, field.A[j], r)
        Ar = _interp(field, field.Ar[j], r)
        B = _interp(field, field.B[j], r)
        Br = _interp(field, field.Br[j], r)
        At = _interp(field, field.At[j], r)
        Bt = _interp(field, field.Bt[j], r)

        Ur = At + v_s * Ar + (2.0 * base_flux / r**3) * A
        Ut = Bt + v_s * Br - v_s * B / r

        vr = vr + Ur * Y
        vr_modal = vr_modal + Ur * Y
        dvr_dth = dvr_dth + Ur * Yt
        dvr_dph = dvr_dph + Ur * Yp

        vth = vth + Ut * Yt
        dvth_dth = dvth_dth + Ut * Ytt
        dvth_dph = dvth_dph + Ut * Ytp

        vph = vph + Ut * (Yp / s_safe)
        dvph_dth = dvph_dth + Ut * (Ytp / s_safe - Yp * np.cos(theta) / s_safe**2)
        dvph_dph = dvph_dph + Ut * (Ypp / s_safe)

    dvr_dr = -2.0 * base_flux / r**3 + _radial_gradient(vr_modal, r)
    dvth_dr = _radial_gradient(vth, r)
    dvph_dr = _radial_gradient(vph, r)

    grr = dvr_dr
    grt = dvr_dth / r - vth / r
    grp = dvr_dph / (r * s_safe) - vph / r
    gtr = dvth_dr
    gtt = vr / r + dvth_dth / r
    gtp = dvth_dph / (r * s_safe) - vph * cot / r
    gpr = dvph_dr
    gpt = dvph_dth / r
    gpp = dvph_dph / (r * s_safe) + vr / r + vth * cot / r

    return (grr, gtt, gpp, 0.5 * (grt + gtr), 0.5 * (grp + gpr), 0.5 * (gtp + gpt))


def _radial_gradient(values, r):
    """Differentiate along the radial axis of a plotting grid."""
    values = np.atleast_2d(values)
    r = np.atleast_2d(r)
    out = np.empty_like(values)
    for k in range(values.shape[1]):
        out[:, k] = np.gradient(values[:, k], r[:, k] if r.shape[1] > 1 else r[:, 0])
    return out


def stress_components(field: ModalField, r_grid, theta, az=None, Rref: float = 1.0,
                      G: float = 1.0, mu: float = 0.0, alpha: float = 0.0,
                      pressure: float | None = None,
                      mode: str = "axisymmetric", part: str = "cos"):
    """Cauchy stress for the quadratic-law Kelvin-Voigt model.

        sigma = G (1 + alpha (I_B - 3)) B + 2 mu D - P I

    ``B`` is recovered from the Almansi strain through ``B^-1 = I - 2e``.
    Omitting ``pressure`` returns the constitutive part, which equals the full
    Cauchy stress for the shear components.
    """
    err, ett, epp, ert, erp, etp = strain_components(
        field, r_grid, theta, az, Rref, "almansi", mode, part
    )
    drr, dtt, dpp, drt, drp, dtp = rate_of_strain_components(field, r_grid, theta, az, mode, part)

    m11, m22, m33 = 1 - 2 * err, 1 - 2 * ett, 1 - 2 * epp
    m12, m13, m23 = -2 * ert, -2 * erp, -2 * etp

    det = (m11 * (m22 * m33 - m23**2) - m12 * (m12 * m33 - m13 * m23)
           + m13 * (m12 * m23 - m13 * m22))
    det = np.where(np.abs(det) < 1e-14, np.nan, det)

    Brr = (m22 * m33 - m23**2) / det
    Btt = (m11 * m33 - m13**2) / det
    Bpp = (m11 * m22 - m12**2) / det
    Brt = (m13 * m23 - m12 * m33) / det
    Brp = (m12 * m23 - m13 * m22) / det
    Btp = (m12 * m13 - m11 * m23) / det

    stiffness = G * (1.0 + alpha * (Brr + Btt + Bpp - 3.0))
    srr = stiffness * Brr + 2 * mu * drr
    stt = stiffness * Btt + 2 * mu * dtt
    spp = stiffness * Bpp + 2 * mu * dpp
    srt = stiffness * Brt + 2 * mu * drt
    srp = stiffness * Brp + 2 * mu * drp
    stp = stiffness * Btp + 2 * mu * dtp

    if pressure is not None:
        srr, stt, spp = srr - pressure, stt - pressure, spp - pressure
    return srr, stt, spp, srt, srp, stp


# --------------------------------------------------------------------------
# Reducing a tensor to a plottable scalar
# --------------------------------------------------------------------------

SCALAR_ALIASES = {
    "err": "err", "radial": "err", "drr": "err", "srr": "err",
    "ett": "ett", "theta": "ett", "dtt": "ett", "stt": "ett",
    "epp": "epp", "phi": "epp", "dpp": "epp", "spp": "epp",
    "ert": "ert", "shear": "ert", "rtheta": "ert", "drt": "ert", "srt": "ert",
    "erp": "erp", "rphi": "erp", "drp": "erp", "srp": "erp",
    "etp": "etp", "thetaphi": "etp", "dtp": "etp", "stp": "etp",
}


def tensor_scalar(components, choice: str = "ert"):
    """Reduce symmetric tensor ``components`` to one scalar field.

    ``components`` is ``(crr, ctt, cpp, crt, crp, ctp)``.  ``choice`` names a
    component (with the aliases in :data:`SCALAR_ALIASES`) or an invariant:
    ``signed_maxabs_principal``, ``max_principal``, ``min_principal``,
    ``maxabs_component``, ``frobenius`` or ``trace``.
    """
    crr, ctt, cpp, crt, crp, ctp = components
    key = str(choice).lower()

    if key in SCALAR_ALIASES:
        return {"err": crr, "ett": ctt, "epp": cpp,
                "ert": crt, "erp": crp, "etp": ctp}[SCALAR_ALIASES[key]]

    if key in ("frobenius", "norm", "magnitude"):
        return np.sqrt(crr**2 + ctt**2 + cpp**2 + 2 * (crt**2 + crp**2 + ctp**2))
    if key in ("trace", "volumetric", "dilatation"):
        return crr + ctt + cpp
    if key in ("maxabs_component", "max_abs_component", "component_max"):
        return np.maximum.reduce([np.abs(c) for c in components])

    principal = {"signed_maxabs_principal": "signed_maxabs",
                 "principal_signed": "signed_maxabs",
                 "max_principal": "max", "largest_principal": "max",
                 "min_principal": "min", "smallest_principal": "min"}
    if key in principal:
        return _principal(components, principal[key])

    raise ValueError(f"unknown scalar choice {choice!r}")


def _principal(components, which: str):
    """Eigenvalue of the symmetric tensor selected by ``which``."""
    crr, ctt, cpp, crt, crp, ctp = (np.asarray(c, dtype=float) for c in components)
    shape = np.broadcast(crr, ctt, cpp, crt, crp, ctp).shape
    M = np.empty(shape + (3, 3))
    M[..., 0, 0], M[..., 1, 1], M[..., 2, 2] = np.broadcast_arrays(crr, ctt, cpp)
    M[..., 0, 1] = M[..., 1, 0] = np.broadcast_to(crt, shape)
    M[..., 0, 2] = M[..., 2, 0] = np.broadcast_to(crp, shape)
    M[..., 1, 2] = M[..., 2, 1] = np.broadcast_to(ctp, shape)

    finite = np.all(np.isfinite(M), axis=(-2, -1))
    eig = np.full(shape + (3,), np.nan)
    if np.any(finite):
        eig[finite] = np.linalg.eigvalsh(M[finite])

    if which == "max":
        return eig[..., -1]
    if which == "min":
        return eig[..., 0]
    # Signed value of largest magnitude.
    idx = np.argmax(np.abs(np.nan_to_num(eig)), axis=-1)
    return np.take_along_axis(eig, idx[..., None], axis=-1)[..., 0]
