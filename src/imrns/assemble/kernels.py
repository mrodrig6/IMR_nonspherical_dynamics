"""Coefficient kernels for the linearised nonspherical perturbation equations.

Every function here is a transliteration of the corresponding hand-expanded
MATLAB expression, preserved term for term so the two can be compared directly.
The source file for each is named in its docstring.

Two deliberate, mechanical departures from the MATLAB source:

1. **Reciprocals are passed in.**  Where MATLAB writes ``X./Re`` these functions
   write ``X * inv_Re``.  The dimensionless groups become infinite in the
   inviscid (``mu = 0``) and purely viscous (``G = 0``) limits, and arithmetic
   on infinities is undefined under Numba's ``fastmath``.  The caller computes
   ``inv_Re = 0.0`` for ``Re = inf`` once, and everything downstream is finite.

2. **Special functions are passed in as scalars.**  ``2F1`` appears only in
   per-mode scalar coefficients -- five distinct values per mode per timestep.
   The backend evaluates them (``scipy.special.hyp2f1``, or a ``custom_jvp``
   wrapper under JAX) and hands them over as plain numbers.

Together these leave every kernel as pure arithmetic on arrays and scalars,
which is simultaneously valid NumPy, Numba-``njit``-able, and JAX-traceable.
That is why there is one copy of the physics and not three.

Symbols
-------
``r``    radial coordinate on the current grid
``rs``   reference radius of the material now at ``r``: ``(r^3-R^3+Req^3)^(1/3)``
``vs``   displaced volume ``Req^3 ((R/Req)^3 - 1)``
``sr``   wall stretch ``R/Req``
``n``    spherical-harmonic mode number
``alph`` second-order (Yeoh) coefficient of the strain-energy function
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Local (non-integral) coefficients of the rotational T equation
# --------------------------------------------------------------------------


def non_int_coeffs(r, R, Rd, Rdd, n, inv_Ca, inv_Re, alph, rs, vs):
    """Pointwise coefficients multiplying derivatives of ``T`` and ``V``.

    From ``matlab_legacy/common/f_non_int_coeffs.m``.  The returned tuple is
    ``(Ct1, ..., Ct7)``; :func:`imrns.assemble.numpy_backend.apply_non_int_terms`
    documents which derivative each one multiplies.

    ``Ct2`` is returned as a scalar, matching the MATLAB source.
    """
    Ct1 = -((R**2 * Rd) / r**2 * inv_Re)

    Ct2 = -inv_Re

    Ct3 = (
        r**4 * R**2 * Rd * (R**2 * Rd + (8.0 * r) * inv_Re)
        + ((-1.0 + 3.0 * alph) * r**4 * rs**4) * inv_Ca
        - (alph * rs**2 * (3.0 * r**6 - 2.0 * r**3 * vs + vs**2)) * inv_Ca
    ) / r**8

    Ct4 = (2.0 * R**2 * Rd) / r**2

    Ct5 = (
        r**4
        * R
        * (r**3 * R * Rdd + Rd * (2.0 * (r**3 - 4.0 * R**3) * Rd + ((-36.0 + n + n**2) * r * R) * inv_Re))
        - (
            8.0
            * vs
            * (3.0 * alph * r**6 + (1.0 - 3.0 * alph) * r**4 * rs**2 - 4.0 * alph * r**3 * vs + 2.0 * alph * vs**2)
        )
        * inv_Ca
        / rs
    ) / r**9

    Ct6 = (-6.0 * R**2 * Rd + (n * (1.0 + n) * r) * inv_Re) / r**3

    Ct7 = (
        2.0
        * r**4
        * R
        * (-2.0 * r**3 * R * Rdd + Rd * ((-4.0 * r**3 + 13.0 * R**3) * Rd + ((48.0 + n + n**2) * r * R) * inv_Re))
        * rs**4
        - ((-1.0 + 3.0 * alph) * r**4 * rs**2 * (n * (1.0 + n) * r**6 + 20.0 * r**3 * vs - 26.0 * vs**2)) * inv_Ca
        + (
            alph
            * (
                3.0 * n * (1.0 + n) * r**12
                - 2.0 * (-5.0 + n) * (6.0 + n) * r**9 * vs
                + (-190.0 + 9.0 * n * (1.0 + n)) * r**6 * vs**2
                - 8.0 * (-27.0 + n + n**2) * r**3 * vs**3
                + 2.0 * (-43.0 + n + n**2) * vs**4
            )
        )
        * inv_Ca
    ) / (r**10 * rs**4)

    return Ct1, Ct2, Ct3, Ct4, Ct5, Ct6, Ct7


# --------------------------------------------------------------------------
# Integral-kernel coefficients
# --------------------------------------------------------------------------


def kappa_coeffs(r, R, Rd, Rdd, n, inv_Ca, inv_Re, alph, rs, vs):
    """Coefficients of the ``kappa`` integral term in the ``V`` rows.

    From ``matlab_legacy/common/f_bulk_kappa.m``.  Returns ``(Ckap, Ckapdot)``,
    which multiply the kappa quadrature row acting on ``T`` and on ``V``.
    """
    Ckap = (
        -(
            (
                2.0
                * n
                * (2.0 + n)
                * r ** (-12.0 - n)
                * (
                    3.0 * (7.0 + n) * r**4 * R ** (5.0 + 2.0 * n) * Rd**2 * rs**7
                    - 3.0 * r**7 * R ** (2.0 + 2.0 * n) * ((3.0 + 2.0 * n) * Rd**2 + R * Rdd) * rs**7
                )
            )
            / ((1.0 + n) * rs**7)
        )
        - (60.0 * n * (2.0 + n) * (3.0 + n) * r ** (-7.0 - n) * R**2 * Rd * R ** (1.0 + 2.0 * n))
        / (1.0 + n)
        * inv_Re
        - inv_Ca
        / ((1.0 + n) * rs**7)
        * 2.0
        * n
        * (2.0 + n)
        * r ** (-12.0 - n)
        * R ** (1.0 + 2.0 * n)
        * vs
        * (
            r**4 * rs**2 * (5.0 * (3.0 + n) * r**6 - (37.0 + 7.0 * n) * r**3 * vs + 3.0 * (7.0 + n) * vs**2)
            + alph
            * (
                15.0 * (3.0 + n) * r**10 * (r - rs) * (r + rs)
                + r**7 * ((-177.0 + n + 4.0 * n**2) * r**2 + 3.0 * (37.0 + 7.0 * n) * rs**2) * vs
                - r**4 * (8.0 * (-36.0 + n * (3.0 + n)) * r**2 + 9.0 * (7.0 + n) * rs**2) * vs**2
                + (-222.0 + n * (21.0 + 5.0 * n)) * r**3 * vs**3
                - (-6.0 + n) * (11.0 + n) * vs**4
            )
        )
    )

    Ckapdot = (6.0 * n * (2.0 + n) * r ** (-5.0 - n) * R ** (3.0 + 2.0 * n) * Rd) / (1.0 + n)

    return Ckap, Ckapdot


def KJ_coeffs(r, rs, vs, n, R, Rd, Rdd, inv_Ca, inv_Re, alph):
    """Coefficients of the ``J`` and ``K`` integral terms in the ``V`` rows.

    From ``matlab_legacy/common/f_bulk_KJ.m``.  Returns ``(CJ, CK, CJd, CKd)``.
    """
    CJ = (
        -((60.0 * n * (2.0 + n) * (3.0 + n) * r ** (-7.0 - n) * R**2 * Rd) / (1.0 + 2.0 * n) * inv_Re)
        - (
            2.0
            * n
            * (2.0 + n)
            * r ** (-12.0 - n)
            * (
                3.0 * (7.0 + n) * r**4 * R**4 * Rd**2 * rs**7
                - 3.0 * r**7 * R * (2.0 * Rd**2 + R * Rdd) * rs**7
            )
        )
        / ((1.0 + 2.0 * n) * rs**7)
        - inv_Ca
        / ((1.0 + 2.0 * n) * rs**7)
        * 2.0
        * n
        * (2.0 + n)
        * r ** (-12.0 - n)
        * vs
        * (
            r**4 * rs**2 * (5.0 * (3.0 + n) * r**6 - (37.0 + 7.0 * n) * r**3 * vs + 3.0 * (7.0 + n) * vs**2)
            + alph
            * (
                15.0 * (3.0 + n) * r**10 * (r - rs) * (r + rs)
                + r**7 * ((-177.0 + n + 4.0 * n**2) * r**2 + 3.0 * (37.0 + 7.0 * n) * rs**2) * vs
                - r**4 * (8.0 * (-36.0 + n * (3.0 + n)) * r**2 + 9.0 * (7.0 + n) * rs**2) * vs**2
                + (-222.0 + n * (21.0 + 5.0 * n)) * r**3 * vs**3
                - (-6.0 + n) * (11.0 + n) * vs**4
            )
        )
    )

    CK = (
        -((60.0 * (-2.0 + n) * (-1.0 + n**2) * r ** (-6.0 + n) * R**2 * Rd) / (1.0 + 2.0 * n) * inv_Re)
        - (
            2.0
            * (-1.0 + n**2)
            * r ** (-11.0 + n)
            * (
                3.0 * (-6.0 + n) * r**4 * R**4 * Rd**2 * rs**7
                + 3.0 * r**7 * R * (2.0 * Rd**2 + R * Rdd) * rs**7
            )
        )
        / ((1.0 + 2.0 * n) * rs**7)
        - inv_Ca
        / ((1.0 + 2.0 * n) * rs**7)
        * 2.0
        * (-1.0 + n**2)
        * r ** (-11.0 + n)
        * vs
        * (
            r**4 * rs**2 * (5.0 * (-2.0 + n) * r**6 + (30.0 - 7.0 * n) * r**3 * vs + 3.0 * (-6.0 + n) * vs**2)
            + alph
            * (
                15.0 * (-2.0 + n) * r**10 * (r - rs) * (r + rs)
                + r**7 * ((174.0 - n * (7.0 + 4.0 * n)) * r**2 + 3.0 * (-30.0 + 7.0 * n) * rs**2) * vs
                + r**4 * (8.0 * (-38.0 + (-1.0 + n) * n) * r**2 - 9.0 * (-6.0 + n) * rs**2) * vs**2
                + (238.0 + (11.0 - 5.0 * n) * n) * r**3 * vs**3
                + (-10.0 + n) * (7.0 + n) * vs**4
            )
        )
    )

    CJd = (6.0 * n * (2.0 + n) * r ** (-5.0 - n) * R**2 * Rd) / (1.0 + 2.0 * n)

    CKd = -((6.0 * (-1.0 + n**2) * r ** (-4.0 + n) * R**2 * Rd) / (1.0 + 2.0 * n))

    return CJ, CK, CJd, CKd


def Teq_epsilon_coeffs(R, Rd, Rdd, r, n, inv_Ca, alph, inv_Re, rs, vs):
    """Coefficients coupling ``eps`` and ``eps'`` into the ``V`` rows.

    From ``matlab_legacy/common/f_Teq_epsilon_coefs.m``.  Returns
    ``(ep_coef, epd_coef)``, both arrays over ``r``.
    """
    ep_coef = -(
        (
            2.0
            * (2.0 + n)
            * r ** (-12.0 - n)
            * R ** (3.0 + n)
            * (
                3.0
                * r**4
                * R
                * (
                    (7.0 + n) * R**3 * Rd**2
                    - r**3 * ((5.0 + n) * Rd**2 + R * Rdd)
                    + (10.0 * (3.0 + n) * r * R * Rd) * inv_Re
                )
                * rs**7
                + (
                    vs
                    * (
                        -(
                            (-1.0 + 3.0 * alph)
                            * r**4
                            * rs**2
                            * (5.0 * (3.0 + n) * r**6 - (37.0 + 7.0 * n) * r**3 * vs + 3.0 * (7.0 + n) * vs**2)
                        )
                        + alph
                        * (
                            15.0 * (3.0 + n) * r**12
                            + (-177.0 + n + 4.0 * n**2) * r**9 * vs
                            - 8.0 * (-36.0 + n * (3.0 + n)) * r**6 * vs**2
                            + (-222.0 + n * (21.0 + 5.0 * n)) * r**3 * vs**3
                            - (-6.0 + n) * (11.0 + n) * vs**4
                        )
                    )
                )
                * inv_Ca
            )
        )
        / ((1.0 + n) * rs**7)
    )

    epd_coef = (6.0 * (2.0 + n) * r ** (-5.0 - n) * R ** (5.0 + n) * Rd) / (1.0 + n)

    return ep_coef, epd_coef


# --------------------------------------------------------------------------
# Interface-amplitude equation
# --------------------------------------------------------------------------


def epsilon_coeffs_me(R, U, lam, Rddot, n, Ro, inv_We, inv_Re, inv_Ca, alpha, F):
    """Stiffness ``xi`` and damping ``eta`` of the ``eps`` equation, model ``"me"``.

    From the ``mod == "me"`` branch of
    ``matlab_legacy/common/f_epsilon_coeffs.m``.  The amplitude obeys
    ``eps'' = -xi eps - eta eps'``, so ``xi`` acts as a stiffness and ``eta`` as
    a damping rate.

    ``lam`` is the wall stretch ``R/Req`` and ``Ro`` is ``Req``.  ``F`` holds
    the precomputed ``2F1`` values; see :func:`hyp_arguments`.
    """
    f_m13, f_p23, f_p13, f_m23, _f_one = F

    eta = 4.0 * (5.0 + 2.0 * n) * U / ((4.0 + n) * R)
    xi = 6.0 * (n + 2.0) / (n + 4.0) * U**2 / R**2 + (8.0 + n - n**2) * Rddot / ((n + 4.0) * R)

    # First-order elastic contribution (the coefficient of G).
    elast = (
        1.0
        / (lam**11 * (6.0 + n) * (9.0 + n) * (12.0 + n) * Ro**2)
        * (1.0 + n)
        * (
            2.0 * lam**5 * (1.0 + 2.0 * lam**3 - lam**6 + n) * (648.0 + 234.0 * n + 27.0 * n**2 + n**3)
            + lam**3
            * (-1.0 + lam**3)
            * (13.0 + n)
            * (
                -n * (14.0 + 9.0 * n + n**2)
                + 2.0 * lam**6 * (54.0 + 15.0 * n + n**2)
                + lam**3 * n * (86.0 + 35.0 * n + 3.0 * n**2)
            )
            * f_m13
            + (-1.0 + lam**3)
            * (
                -2.0 * lam**9 * (54.0 + 15.0 * n + n**2)
                + lam**6 * n * (54.0 + 69.0 * n + 16.0 * n**2 + n**3)
                + n * (140.0 + 104.0 * n + 19.0 * n**2 + n**3)
                - lam**3 * n * (266.0 + 199.0 * n + 37.0 * n**2 + 2.0 * n**3)
            )
            * f_p23
        )
    )

    # Second-order elastic contribution (the coefficient of alpha G).
    sselast = (1.0 / (3.0 * lam**10 * (6.0 + n) * (9.0 + n) * Ro**2)) * (1.0 + n) * (
        6.0
        * (1.0 - lam**2)
        * (6.0 + n)
        * (9.0 + n)
        * (
            1.0
            + lam**2
            - lam**3 * (-4.0 + lam * (2.0 + lam * (-4.0 + lam + 2.0 * lam**2 + lam**3 - 2.0 * lam**5)))
            - (-2.0 - 2.0 * lam**2 + lam**4) * n
        )
        - (1.0 / (lam * (12.0 + n)))
        * 3.0
        * (1.0 - lam**3)
        * (
            lam**3
            * (
                (14.0 + n)
                * (
                    4.0 * lam**9 * (6.0 + n) * (9.0 + n)
                    + 2.0 * lam**6 * (1.0 + n) * (2.0 + n) * (6.0 + n) * (9.0 + n)
                    + n * (318.0 + n * (175.0 + n * (33.0 + 2.0 * n)))
                    - lam**3 * (-324.0 + n * (120.0 + n * (229.0 + n * (59.0 + 4.0 * n))))
                )
                * f_m23
                + 3.0
                * (13.0 + n)
                * (
                    n * (2.0 + n) * (7.0 + n)
                    - 2.0 * lam**6 * (6.0 + n) * (9.0 + n)
                    - lam**3 * n * (86.0 + n * (35.0 + 3.0 * n))
                )
                * f_m13
            )
            - 2.0
            * (
                4.0 * lam**12 * (6.0 + n) * (9.0 + n)
                + 2.0 * lam**9 * (1.0 + n) * (2.0 + n) * (6.0 + n) * (9.0 + n)
                + lam**6 * (1.0 + n) * (6.0 + n) * (9.0 + n) * (6.0 + n * (6.0 + n))
                + n * (2.0 + n) * (11.0 + n) * (39.0 + n * (13.0 + n))
                - lam**3 * n * (1398.0 + n * (1411.0 + n * (427.0 + 2.0 * n * (25.0 + n))))
            )
            * f_p13
            - 3.0
            * (
                -2.0 * lam**9 * (6.0 + n) * (9.0 + n)
                + lam**6 * n * (1.0 + n) * (6.0 + n) * (9.0 + n)
                + n * (2.0 + n) * (7.0 + n) * (10.0 + n)
                - lam**3 * n * (2.0 + n) * (7.0 + n) * (19.0 + 2.0 * n)
            )
            * f_p23
        )
    )

    visceta = 2.0 * (n + 1.0) * (n + 2.0) / R**2
    viscxi = (n * (n + 1.0) * (20.0 + 7.0 * n)) * U / ((6.0 + n) * R**3)
    surften = (n + 1.0) * (n - 1.0) * (n + 2.0) / R**3

    xi = xi + inv_Ca * elast + alpha * inv_Ca * sselast + 0.5 * inv_We * surften + inv_Re * viscxi
    eta = eta + inv_Re * visceta

    return xi, eta


def epsilon_coeffs_pros(R, U, lam, Rddot, n, inv_We, inv_Re, inv_Ca):
    """Stiffness and damping of the ``eps`` equation, model ``"pros"``.

    From the ``mod == "Pros"`` branch of
    ``matlab_legacy/common/f_epsilon_coeffs.m``.
    """
    adot_prefact = 3.0 * U / R + 2.0 / R**2 * (n + 2.0) * (n + 1.0) * inv_Re
    aprefact = (
        -(n - 1.0) * Rddot / R
        + 4.0 * (n - 1.0) * (n + 1.0) * U * inv_Re / R**3
        + (n - 1.0) * (n + 2.0) * (n + 1.0) * 0.5 * inv_We / R**3
        + inv_Ca / R**2 * (2.0 * lam ** (-1.0) + 2.0 * lam ** (-4.0) + n * (n + 1.0) / (1.0 + lam ** (-1.0) + lam ** (-2.0)))
    )

    xi = aprefact + adot_prefact * U / R + Rddot / R
    eta = adot_prefact + 2.0 * U / R

    return xi, eta


# --------------------------------------------------------------------------
# Prescribed-amplitude source terms for the T equation
# --------------------------------------------------------------------------


def forced_source(ep, epd, R, Rd, Rdd, r, n, inv_Ca, inv_Re, alph, rs, vs):
    """Source entering the ``V`` rows when the amplitude is prescribed.

    From ``matlab_legacy/common/f_source_terms.m``.  Returns ``Sp - Ap``, where
    ``Ap`` collects the contribution of the potential-flow acceleration and
    ``Sp`` that of the potential-flow stress.
    """
    Ap = (
        6.0
        * (n + 2.0)
        / (n + 1.0)
        * R ** (n + 4.0)
        / (r ** (n + 9.0))
        * (
            ep * (Rd**2 * ((n + 5.0) * r**3 - (n + 7.0) * R**3) + r**3 * R * Rdd)
            + r**3 * R * Rd * epd
        )
    )

    Sp = (
        2.0
        * ep
        * (n + 2.0)
        / (n + 1.0)
        * R ** (n + 3.0)
        / (rs**7 * r ** (n + 13.0))
        * (
            (30.0 * (3.0 + n) * r**5 * Rd * R**2 * rs**7) * inv_Re
            + inv_Ca
            * vs
            * (
                -(
                    (-1.0 + 3.0 * alph)
                    * r**4
                    * rs**2
                    * (5.0 * (3.0 + n) * r**6 - (37.0 + 7.0 * n) * r**3 * vs + 3.0 * (7.0 + n) * vs**2)
                )
                + alph
                * (
                    15.0 * (3.0 + n) * r**12
                    + (-177.0 + n + 4.0 * n**2) * r**9 * vs
                    - 8.0 * (-36.0 + n * (3.0 + n)) * r**6 * vs**2
                    + (-222.0 + n * (21.0 + 5.0 * n)) * r**3 * vs**3
                    - (-6.0 + n) * (11.0 + n) * vs**4
                )
            )
        )
    )

    return Sp - Ap


# --------------------------------------------------------------------------
# Feedback of the rotational field onto the amplitude equation
# --------------------------------------------------------------------------


def ep_feedback_pointwise(r, R, Rd, n, inv_Ca, inv_Re, alph, rs):
    """Integrand coefficients of the ``T``-field feedback on ``eps''``.

    From the ``fTintegrand`` block of
    ``matlab_legacy/common/f_ep_source_terms.m``.  Returns
    ``(fT_PI, fV_PI, fdTdr_PI)``, the weights of ``T``, ``V`` and ``dT/dr``
    inside the radial integral.
    """
    fT_PI = -(
        (
            n
            * (1.0 + n) ** 2
            * (
                -(r**4) * Rd * (R**4 * Rd + (4.0 * r * R**2) * inv_Re)
                + ((1.0 - 3.0 * alph) * r**4 * rs**4) * inv_Ca
                + (3.0 * alph * rs**8) * inv_Ca
            )
        )
        / (r**10 * R**2)
    )
    fV_PI = -((n * (1.0 + n) ** 2) / (r**2 * R**2) * inv_Re)
    fdTdr_PI = -((n * (1.0 + n) ** 2 * Rd) / r**4 * inv_Re)
    return fT_PI, fV_PI, fdTdr_PI


def ep_feedback_kappa(R, Rd, Rdd, sr, n, inv_Ca, inv_Re, alph, F):
    """Scalar kappa coefficients feeding back onto ``eps''``.

    From the kappa block of ``matlab_legacy/common/f_ep_source_terms.m``.
    Returns ``(H_kapddot, G_kapdot, F_kap, fT_wall)``.  ``F`` holds the
    precomputed ``2F1`` values; see :func:`hyp_arguments`.
    """
    f_m13, f_p23, f_p13, _f_m23, f_one = F

    H_kapddot = -n * R ** (n - 2.0)

    G_kapdot = (
        R ** (-4.0 + n)
        * (
            -2.0 * n * (1.0 + n) * (4.0 + n) * (1.0 + 2.0 * n) * inv_Re
            - 2.0 * n * (2.0 + n * (6.0 + n)) * R * Rd
        )
    ) / (4.0 + n)

    F_kap = n * (
        -(((4.0 + n * (8.0 + n)) * R ** (-4.0 + n) * ((-1.0 + n) * Rd**2 + R * Rdd)) / (4.0 + n))
        - 2.0
        * (1.0 + n)
        * (1.0 + 2.0 * n)
        * (R / sr) ** (-5.0 + n)
        * sr ** (-12.0 + n)
        * (
            ((-2.0 + n) * Rd * sr**7) * inv_Re
            + ((R / sr) * (sr**4 + alph * (2.0 - 3.0 * sr**4 + sr**6))) * inv_Ca
        )
        + 1.0
        / ((6.0 + n) * (9.0 + n) * (12.0 + n) * R**2)
        * (1.0 + n)
        * (R / sr) ** (-3.0 + n)
        * sr ** (-11.0 + n)
        * (
            -((6.0 * n * (2.0 + n) * (9.0 + n) * (12.0 + n) * Rd * sr**8) * inv_Re)
            + inv_Ca
            * (R / sr)
            * (-1.0 + sr**3)
            * (
                (-1.0 + 3.0 * alph)
                * (13.0 + n)
                * sr**3
                * (
                    -n * (2.0 + n) * (7.0 + n)
                    + n * (86.0 + n * (35.0 + 3.0 * n)) * sr**3
                    + 2.0 * (6.0 + n) * (9.0 + n) * sr**6
                )
                * f_m13
                + 2.0
                * alph
                * (
                    n * (2.0 + n) * (11.0 + n) * (39.0 + n * (13.0 + n))
                    - n * (1398.0 + n * (1411.0 + n * (427.0 + 2.0 * n * (25.0 + n)))) * sr**3
                    + (1.0 + n) * (6.0 + n) * (9.0 + n) * (6.0 + n * (6.0 + n)) * sr**6
                    + 2.0 * (1.0 + n) * (2.0 + n) * (6.0 + n) * (9.0 + n) * sr**9
                    + 4.0 * (6.0 + n) * (9.0 + n) * sr**12
                )
                * f_p13
                + (-1.0 + 3.0 * alph)
                * (
                    n * (2.0 + n) * (7.0 + n) * (10.0 + n)
                    - n * (2.0 + n) * (7.0 + n) * (19.0 + 2.0 * n) * sr**3
                    + n * (1.0 + n) * (6.0 + n) * (9.0 + n) * sr**6
                    - 2.0 * (6.0 + n) * (9.0 + n) * sr**9
                )
                * f_p23
                - 1.0
                / sr**2
                * alph
                * (14.0 + n)
                * (
                    n * (318.0 + n * (175.0 + n * (33.0 + 2.0 * n)))
                    - (-324.0 + n * (120.0 + n * (229.0 + n * (59.0 + 4.0 * n)))) * sr**3
                    + 2.0 * (1.0 + n) * (2.0 + n) * (6.0 + n) * (9.0 + n) * sr**6
                    + 4.0 * (6.0 + n) * (9.0 + n) * sr**9
                )
                * f_one
            )
        )
    )

    # Single boundary term left over after f_T, G and H are combined.
    fT_wall = -((n * (1.0 + n) * Rd * (R * Rd + (2.0 * (1.0 + n)) * inv_Re)) / R**4)

    return H_kapddot, G_kapdot, F_kap, fT_wall


def ep_feedback_J(r, R, Rd, Rdd, Req, sr, n, inv_Ca, inv_Re, alph, rs, vs):
    """``J``-kernel integrand coefficients feeding back onto ``eps''``.

    From the ``f_J`` block of ``matlab_legacy/common/f_ep_source_terms.m``.
    Returns ``(fJ_T, fJ_dT, fJ_dt2)``.
    """
    fJ_T = (
        1.0
        / (1.0 + 2.0 * n)
        * n
        * (1.0 + n) ** 2
        * r ** (-12.0 - n)
        * (
            (n * r**4 * (2.0 * r**3 - (7.0 + n) * R**3) * Rd**2) / R
            + n * r**7 * Rdd
            - (6.0 * n * (2.0 + n) * r**5 * Rd) * inv_Re
            - inv_Ca
            / (rs**7 * sr**2)
            * Req
            * (-1.0 + sr**3)
            * (
                -(
                    (-1.0 + 3.0 * alph)
                    * r**4
                    * rs**2
                    * (
                        2.0 * (2.0 + n) * (3.0 + n) * r**6
                        - (8.0 + n * (17.0 + 3.0 * n)) * r**3 * vs
                        + n * (7.0 + n) * vs**2
                    )
                )
                + alph
                * (
                    2.0 * (2.0 + n) * (21.0 + 5.0 * n) * r**12
                    - (172.0 + 27.0 * n * (7.0 + n)) * r**9 * vs
                    + (152.0 + n * (235.0 + 29.0 * n)) * r**6 * vs**2
                    - 3.0 * (16.0 + n * (47.0 + 5.0 * n)) * r**3 * vs**3
                    + 3.0 * n * (11.0 + n) * vs**4
                )
            )
        )
    )

    fJ_dT = (2.0 * n * (1.0 + n) ** 2 * (2.0 + n) * r ** (-5.0 - n) * Rd) / (1.0 + 2.0 * n)

    fJ_dt2 = -n * (n + 1.0) ** 2 / ((2.0 * n + 1.0) * R**2) * r ** (-n - 2.0)

    return fJ_T, fJ_dT, fJ_dt2


def ep_feedback_K(r, R, Rd, Rdd, sr, n, inv_Ca, inv_Re, alph, rs, vs):
    """``K``-kernel integrand coefficients feeding back onto ``eps''``.

    From the ``f_K`` block of ``matlab_legacy/common/f_ep_source_terms.m``.
    Returns ``(fK_T, fK_dT, fK_dt2)``.
    """
    fK_T = (
        n
        * (1.0 + n) ** 2
        * r ** (-11.0 + n)
        * (
            ((1.0 + n) * r**4 * ((2.0 * r**3 + (-6.0 + n) * R**3) * Rd**2 + r**3 * R * Rdd))
            / ((1.0 + 2.0 * n) * R)
            + 1.0
            / (1.0 + 2.0 * n)
            * (
                (6.0 * (-1.0 + n**2) * r**5 * Rd) * inv_Re
                + inv_Ca
                / (rs**7 * sr**2)
                * (R / sr)
                * (-1.0 + sr**3)
                * (
                    -(
                        (-1.0 + 3.0 * alph)
                        * r**4
                        * rs**2
                        * (
                            2.0 * (-2.0 + n) * (-1.0 + n) * r**6
                            + (6.0 + (11.0 - 3.0 * n) * n) * r**3 * vs
                            + (-6.0 + n) * (1.0 + n) * vs**2
                        )
                    )
                    + alph
                    * (
                        2.0 * (-1.0 + n) * (-16.0 + 5.0 * n) * r**12
                        + (-10.0 - 27.0 * (-5.0 + n) * n) * r**9 * vs
                        + (-54.0 + n * (-177.0 + 29.0 * n)) * r**6 * vs**2
                        + 3.0 * (26.0 + (37.0 - 5.0 * n) * n) * r**3 * vs**3
                        + 3.0 * (-10.0 + n) * (1.0 + n) * vs**4
                    )
                )
            )
        )
    )

    fK_dT = 2.0 * (n - 1.0) * n * (n + 1.0) ** 2 * r ** (n - 4.0) * Rd / (2.0 * n + 1.0)

    fK_dt2 = n * (n + 1.0) ** 2 / ((2.0 * n + 1.0) * R**2) * r ** (n - 1.0)

    return fK_T, fK_dT, fK_dt2


# --------------------------------------------------------------------------
# Wall boundary condition
# --------------------------------------------------------------------------


def wall_traction(sr, Req, R, Rd, inv_Ca, inv_Re, alph):
    """Integrated elastic wall traction ``zeta`` and the combined wall term ``fT``.

    From ``matlab_legacy/common/f_boundary_conditions.m``.  ``zeta`` is the
    elastic part of the normal traction at ``r = R``; at ``sr = 1`` it reduces
    to ``Req/Ca`` for any ``alph``, which is a useful check.
    """
    zeta = inv_Ca * Req * (2.0 * alph * sr**6 - 3.0 * alph * sr**4 + alph + sr**4)
    fT = (zeta - 4.0 * sr**7 * inv_Re * Rd) / R**2
    return zeta, fT


def wall_kappa_terms(n, R, Rd, sr, zeta, inv_Re):
    """Kappa-related wall coefficients.

    From ``matlab_legacy/common/f_boundary_conditions.m``.  Returns
    ``(f_kappadot, j2_kappa, j2_kappadot)``.
    """
    f_kappadot = (n + 1.0) / (2.0 * n + 1.0) * R ** (-n) * Rd
    j2_kappa = 2.0 * (2.0 * n + 1.0) / (n + 1.0) * R ** (n - 3.0) * (zeta + (n - 5.0) * sr**7 * inv_Re * Rd)
    j2_kappadot = 2.0 * (2.0 * n + 1.0) / (n + 1.0) * R ** (n - 2.0) * sr**7 * inv_Re
    return f_kappadot, j2_kappa, j2_kappadot


# --------------------------------------------------------------------------
# Hypergeometric arguments
# --------------------------------------------------------------------------


def hyp_arguments(n: float, sr: float):
    """Arguments of the five ``2F1`` evaluations a mode needs at wall stretch ``sr``.

    Returns a list of ``(a, b, c, z)`` tuples in the order consumed by
    :func:`epsilon_coeffs_me` and :func:`ep_feedback_kappa`:

    ``a = -1/3, 2/3, 1/3, -2/3`` with ``(b, c) = (4 + n/3, 5 + n/3)``, then
    ``a = 1`` with ``(b, c) = ((17 + n)/3, 5 + n/3)``.  All share
    ``z = 1 - sr^-3``.
    """
    z = 1.0 - 1.0 / sr**3
    b, c = 4.0 + n / 3.0, 5.0 + n / 3.0
    return [
        (-1.0 / 3.0, b, c, z),
        (2.0 / 3.0, b, c, z),
        (1.0 / 3.0, b, c, z),
        (-2.0 / 3.0, b, c, z),
        (1.0, (17.0 + n) / 3.0, c, z),
    ]
