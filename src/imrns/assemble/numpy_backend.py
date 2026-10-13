"""Reference assembly of the per-mode operator block.

This is the readable implementation.  :mod:`imrns.assemble.numba_backend` and
:mod:`imrns.assemble.jax_backend` reproduce it; :mod:`tests.test_backends`
requires all three to agree.

Block diagonality
-----------------
The legacy code built one ``n_modes * block_size`` square matrix per timestep,
but every write it performed landed inside the writing mode's own block -- the
linearised modes do not couple.  This module therefore assembles one block per
mode, and :mod:`imrns.stepper` solves them independently.  For the default
four-mode, ``xN = 256`` case that replaces a 2056-cubed factorisation with four
514-cubed ones, which measures about five times faster.

Ordering note
-------------
:func:`ep_feedback_rows` captures the ``V`` rows *before*
:func:`add_moving_grid_terms` adds the moving-grid advection to them, so the
second-time-derivative substitution uses the uncorrected operator.  That
matches the call order in
``matlab_legacy/common/compute_rotational_perturbation_evolution.m`` and is
intentional -- confirmed with the author.  Do not reorder these two calls.
"""

from __future__ import annotations

import numpy as np

from ..operators import upwind_advection
from ..special import hyp2f1_values
from . import kernels as K

__all__ = [
    "JK_matrices",
    "add_moving_grid_terms",
    "apply_boundary_conditions",
    "apply_boundary_rhs",
    "assemble_block",
    "ep_feedback_rows",
    "hyp2f1_values",
    "kappa_weights",
]


def kappa_weights(r, drdx, n, w):
    """Quadrature row for the ``kappa`` radial kernel.

    From ``matlab_legacy/common/f_kappa_kernel.m``.
    """
    j1 = -(n + 1.0) / (2.0 * n + 1.0) * r ** (-n) * drdx
    return w * j1


def JK_matrices(r, drdx, One_wT, W, n):
    """Quadrature matrices for the ``J`` and ``K`` radial kernels.

    From ``matlab_legacy/common/f_J_K_kernel.m``.  ``K`` integrates from ``r``
    outwards (hence ``One_wT - W``, the tail of the cumulative integral) and
    ``J`` from the wall to ``r``.
    """
    KmT = (One_wT - W) * (-(r ** (-n)) * drdx)[None, :]
    JmT = W * (r ** (n + 1.0) * drdx)[None, :]
    return KmT, JmT


def _scale_rows(v, M):
    """``diag(v) @ M`` without forming the diagonal matrix."""
    return np.asarray(v).reshape(-1, 1) * M


def assemble_block(mode, idx, metrics, state, groups, alph, model, closure, forced_ep, ops, hyp,
                   kern=K, aadv=None):
    """Assemble the operator block ``A`` for one mode.

    Parameters
    ----------
    mode
        Spherical-harmonic mode number ``n``.
    idx
        :class:`imrns.layout.ModeIndices` rebased to this block.
    metrics
        :class:`imrns.grid.GridMetrics` for the current timestep.
    state
        ``(R, Rd, Rdd, Req)`` at the current timestep.
    groups
        ``(inv_Ca, inv_Re, inv_We)``.
    ops
        ``(H1, H2, H3, W, w, One_wT)`` from :mod:`imrns.operators`.
    hyp
        The five precomputed ``2F1`` values for this mode and wall stretch.
    aadv
        Prebuilt moving-grid advection operator.  It is the same for every mode
        at a given timestep, so the caller builds it once; omit to build it here.

    Returns the dense ``block_size`` square array.
    """
    R, Rd, Rdd, Req = state
    inv_Ca, inv_Re, inv_We = groups
    H1, H2, H3, W, w, One_wT = ops
    n = float(mode)
    r, rs, vs, sr = metrics.r, metrics.rs, metrics.vs, metrics.sr
    dxdr, drdx = metrics.dxdr, metrics.drdx

    A = np.zeros((idx.block_size, idx.block_size))
    T, V, ep, epd = idx.T, idx.V, idx.ep, idx.epd

    if closure == "rot" and model == "me":
        Aadv = upwind_advection(metrics.a, metrics.h) if aadv is None else aadv
        A[idx.T_sl, idx.T_sl] += Aadv
        A[idx.T_sl, idx.V_sl] = np.eye(len(V))

    if not forced_ep:
        A[ep, epd] = 1.0

    if model == "me":
        if not forced_ep:
            xi, eta = kern.epsilon_coeffs_me(R, Rd, sr, Rdd, n, Req, inv_We, inv_Re, inv_Ca, alph, hyp)
            A[epd, ep] -= xi
            A[epd, epd] -= eta
            if closure == "rot":
                ep_c, epd_c = kern.Teq_epsilon_coeffs(R, Rd, Rdd, r, n, inv_Ca, alph, inv_Re, rs, vs)
                A[V, ep] -= ep_c
                A[V, epd] -= epd_c

        if closure == "rot":
            Ct = kern.non_int_coeffs(r, R, Rd, Rdd, n, inv_Ca, inv_Re, alph, rs, vs)
            _add_non_int_terms(A, Ct, metrics, H1, H2, H3, idx)

            wkap = kappa_weights(r, drdx, n, w)
            Ckap, Ckapdot = kern.kappa_coeffs(r, R, Rd, Rdd, n, inv_Ca, inv_Re, alph, rs, vs)
            A[idx.V_sl, idx.T_sl] -= np.outer(Ckap, wkap)
            A[idx.V_sl, idx.V_sl] -= np.outer(Ckapdot, wkap)

            KmT, JmT = JK_matrices(r, drdx, One_wT, W, n)
            CJ, CK, CJd, CKd = kern.KJ_coeffs(r, rs, vs, n, R, Rd, Rdd, inv_Ca, inv_Re, alph)
            A[idx.V_sl, idx.T_sl] -= _scale_rows(CK, KmT) + _scale_rows(CJ, JmT)
            A[idx.V_sl, idx.V_sl] -= _scale_rows(CKd, KmT) + _scale_rows(CJd, JmT)

    elif model == "pros":
        if closure == "rot":
            diag_coef = inv_Re * n * (n + 1.0) / r**2
            col_coef = Rd * R**2 / r**2
            A[idx.T_sl, idx.T_sl] += (
                inv_Re * _scale_rows(metrics.d2xdr2_1, H1)
                + inv_Re * _scale_rows(metrics.d2xdr2_2, H2)
                - _scale_rows(dxdr, H1) * col_coef[None, :]
            )
            A[T, T] -= diag_coef
        if not forced_ep:
            xi, eta = kern.epsilon_coeffs_pros(R, Rd, sr, Rdd, n, inv_We, inv_Re, inv_Ca)
            A[epd, ep] -= xi
            A[epd, epd] -= eta

    return A


def _add_non_int_terms(A, Ct, metrics, H1, H2, H3, idx):
    """Add the local (non-integral) derivative terms to the ``V`` rows.

    From ``matlab_legacy/common/f_mod_A_nonint_terms.m``.  Each coefficient
    multiplies one derivative of the state, converted from ``x`` to ``r`` by the
    chain-rule factors in ``metrics``:

    =======  ====================================================
    ``Ct7``  ``T``                 (pointwise)
    ``Ct5``  ``dT/dr``
    ``Ct3``  ``d2T/dr2``
    ``Ct1``  ``d3T/dr3``
    ``Ct6``  ``V = dT/dt|_r``      (pointwise)
    ``Ct4``  ``d/dr (V)``
    ``Ct2``  ``d2/dr2 (V)``
    =======  ====================================================
    """
    Ct1, Ct2, Ct3, Ct4, Ct5, Ct6, Ct7 = Ct
    T, V = idx.T, idx.V
    VT = (idx.V_sl, idx.T_sl)
    VV = (idx.V_sl, idx.V_sl)

    A[V, T] -= Ct7
    A[VT] -= _scale_rows(Ct5 * metrics.dxdr, H1)
    A[VT] -= _scale_rows(Ct3 * metrics.d2xdr2_2, H2)
    A[VT] -= _scale_rows(Ct3 * metrics.d2xdr2_1, H1)
    A[VT] -= _scale_rows(Ct1 * metrics.d3xdr3_3, H3)
    A[VT] -= _scale_rows(Ct1 * metrics.d3xdr3_2, H2)
    A[VT] -= _scale_rows(Ct1 * metrics.d3xdr3_1, H1)

    A[V, V] -= Ct6
    A[VV] -= _scale_rows(Ct4 * metrics.dxdr, H1)
    A[VV] -= _scale_rows(Ct2 * metrics.d2xdr2_2, H2)
    A[VV] -= _scale_rows(Ct2 * metrics.d2xdr2_1, H1)


def ep_feedback_rows(A, mode, idx, metrics, state, groups, alph, model, ops, hyp, kern=K):
    """Add the rotational field's feedback onto the ``eps''`` row.

    From ``matlab_legacy/common/f_ep_source_terms.m``.  Modifies ``A`` in place.

    See the module docstring on ordering: this must run *before*
    :func:`add_moving_grid_terms`.
    """
    R, Rd, Rdd, Req = state
    inv_Ca, inv_Re, _inv_We = groups
    H1, _H2, _H3, W, w, One_wT = ops
    n = float(mode)
    r, rs, vs, sr = metrics.r, metrics.rs, metrics.vs, metrics.sr
    drdx, dxdr = metrics.drdx, metrics.dxdr
    T, epd = idx.T, idx.epd

    if model == "me":
        quad = w * drdx  # outer radial quadrature row, shared by every term

        fT_PI, fV_PI, fdT_PI = kern.ep_feedback_pointwise(r, R, Rd, n, inv_Ca, inv_Re, alph, rs)
        A[epd, idx.T_sl] += quad * fT_PI + (quad * fdT_PI * dxdr) @ H1
        A[epd, idx.V_sl] += quad * fV_PI

        KmT, JmT = JK_matrices(r, drdx, One_wT, W, n)
        wkap = kappa_weights(r, drdx, n, w)

        # The V rows as they stand now.  Used to express d2T/dt2 in terms of the
        # block state; see the module docstring.
        Li = A[idx.V_sl, :].copy()

        H_kapddot, G_kapdot, F_kap, fT_wall = kern.ep_feedback_kappa(
            R, Rd, Rdd, sr, n, inv_Ca, inv_Re, alph, hyp
        )
        A[epd, :] += H_kapddot * (wkap @ Li)
        A[epd, idx.V_sl] += G_kapdot * wkap
        A[epd, idx.T_sl] += F_kap * wkap
        A[epd, T[0]] += fT_wall

        fJ_T, fJ_dT, fJ_dt2 = kern.ep_feedback_J(r, R, Rd, Rdd, Req, sr, n, inv_Ca, inv_Re, alph, rs, vs)
        A[epd, idx.T_sl] += (quad * fJ_T) @ JmT
        A[epd, idx.V_sl] += (quad * fJ_dT) @ JmT
        A[epd, :] += ((quad * fJ_dt2) @ JmT) @ Li

        fK_T, fK_dT, fK_dt2 = kern.ep_feedback_K(r, R, Rd, Rdd, sr, n, inv_Ca, inv_Re, alph, rs, vs)
        A[epd, idx.T_sl] += (quad * fK_T) @ KmT
        A[epd, idx.V_sl] += (quad * fK_dT) @ KmT
        A[epd, :] += ((quad * fK_dt2) @ KmT) @ Li

    elif model == "pros":
        A[epd, T[0]] -= n * (n + 2.0) * (n + 1.0) * inv_Re / R**3
        A[epd, idx.T_sl] += (
            n * (n + 1.0) * Rd / R**3 * w * drdx * dxdr * (1.0 - (R / r) ** 3) * (R / r) ** n
        )
        Brot = 2.0 * n * (n - 1.0) * (n + 1.0) * Rd * inv_Re / R**3 - n * (n + 1.0) ** 2 * inv_Ca / R**2 * sr ** (-4)
        Arot = -2.0 * n * (n + 2.0) * (n + 1.0) * inv_Re / R**2
        A[idx.epd, idx.ep] -= Brot + Arot * Rd / R
        A[idx.epd, idx.epd] -= Arot

    return A


def add_moving_grid_terms(A, idx, metrics, model, aadv=None):
    """Add the moving-grid advection correction.

    From ``matlab_legacy/common/f_modify_A_bulk_map.m``.  The map ``r(x, R(t))``
    moves with the wall, so ``d/dt|_x`` and ``d/dt|_r`` differ by a transport
    term ``a d/dx`` -- the ALE identity, see :func:`imrns.grid.grid_velocity`.
    Applied once per state variable: :func:`assemble_block` puts it on the ``T``
    rows (where it transforms the definition of ``V``), and this puts it on the
    rows that carry ``V``'s own evolution.  Coefficient 1 in both places.

    For model ``"pros"``, which has no separate ``V``, it lands on ``T``.

    Note for anyone comparing against the MATLAB: that file's docstring claims
    four terms, ``(a_t - a a_x) T_x + 2a V_x - a^2 T_xx``.  That is correct
    mathematics for the superseded convention ``V = dT/dt|_x``; under the
    implemented convention ``V = dT/dt|_r`` three of them collapse and only
    ``a V_x`` remains.  ``docs/moving-grid.md`` works this through, and
    ``tests/test_moving_grid.py`` verifies the two forms are the same equation.
    """
    Aadv = upwind_advection(metrics.a, metrics.h) if aadv is None else aadv
    target = idx.V_sl if model == "me" else idx.T_sl
    A[target, target] += Aadv
    return A


def apply_boundary_conditions(Alhs, rhs, mode, idx, metrics, state, groups, alph, model,
                              forced_ep, ops, ep_now, epd_now, kern=K):
    """Overwrite the wall and far-field rows of the left-hand-side matrix.

    From ``matlab_legacy/common/f_boundary_conditions.m``.  These act on the
    assembled ``beta0 I - dt A``, not on ``A`` itself.

    At the wall the condition is continuity of tangential traction; in the far
    field the rotational field is required to vanish.  Returns ``(Alhs, rhs)``,
    both modified in place.
    """
    R, Rd, _Rdd, Req = state
    inv_Ca, inv_Re, _inv_We = groups
    H1, _H2, _H3, _W, w, _One_wT = ops
    n = float(mode)
    sr, r, drdx, dxdr = metrics.sr, metrics.r, metrics.drdx, metrics.dxdr
    T, V, ep, epd = idx.T, idx.V, idx.ep, idx.epd

    zeta, fT = kern.wall_traction(sr, Req, R, Rd, inv_Ca, inv_Re, alph)
    f_kappadot, j2_kappa, j2_kappadot = kern.wall_kappa_terms(n, R, Rd, sr, zeta, inv_Re)
    wkap = kappa_weights(r, drdx, n, w)

    if model == "me":
        v0 = V[0]
        Alhs[v0, :] = 0.0
        Alhs[v0, v0] += sr**7 / R * inv_Re
        Alhs[v0, idx.T_sl] += sr**7 / R * inv_Re * Rd * dxdr[0] * H1[0, :]
        Alhs[v0, T[0]] += fT
        Alhs[v0, T[0]] -= j2_kappadot * f_kappadot
        Alhs[v0, idx.T_sl] -= j2_kappa * wkap
        Alhs[v0, idx.V_sl] -= j2_kappadot * wkap
        rhs[v0] = 0.0

        # Far field: both T and V are driven to zero.
        Alhs[T[-1], :] = 0.0
        Alhs[T[-1], T[-1]] = 1.0
        rhs[T[-1]] = 0.0
        Alhs[V[-1], :] = 0.0
        Alhs[V[-1], V[-1]] = 1.0
        rhs[V[-1]] = 0.0

        forcing = 2.0 * (n + 2.0) / (n + 1.0) / R
        if forced_ep:
            rhs[v0] += forcing * (
                ep_now * (zeta - 3.0 * sr**7 * inv_Re * Rd) + sr**7 * inv_Re * R * epd_now
            )
        else:
            Alhs[v0, ep] -= forcing * (zeta - 3.0 * sr**7 * inv_Re * Rd)
            Alhs[v0, epd] -= forcing * (sr**7 * inv_Re * R)

    elif model == "pros":
        t0 = T[0]
        Alhs[t0, :] = 0.0
        Alhs[t0, t0] += (n + 1.0) / R * inv_Re
        Alhs[t0, idx.T_sl] -= 2.0 * (2.0 * n + 1.0) * R ** (n - 2.0) * inv_Re * wkap

        Alhs[T[-1], :] = 0.0
        Alhs[T[-1], T[-1]] = 1.0
        rhs[T[-1]] = 0.0

        if forced_ep:
            rhs[t0] = (
                2.0 * (n + 2.0) * inv_Re * (epd_now * R + ep_now * Rd) / R
                - 2.0 * (n - 1.0) * inv_Re * ep_now * Rd / R
            )
        else:
            Alhs[t0, ep] += (
                -2.0 * (n + 2.0) * inv_Re * Rd / R
                + 2.0 * (n - 1.0) * inv_Re * Rd / R
                - (n + 1.0) * inv_Ca * sr ** (-4)
            )
            Alhs[t0, epd] -= 2.0 * (n + 2.0) * inv_Re
            rhs[t0] = 0.0

    return Alhs, rhs



def apply_boundary_rhs(rhs, mode, idx, metrics, state, groups, alph, model, forced_ep,
                       ep_now, epd_now, kern=K):
    """Apply only the right-hand-side part of the boundary conditions.

    Used when the operator is reused from an earlier step: the matrix rows are
    already correct and only the forcing changes.  Mirrors the ``rhs`` writes in
    :func:`apply_boundary_conditions` exactly.
    """
    R, Rd, _Rdd, Req = state
    inv_Ca, inv_Re, _inv_We = groups
    n = float(mode)
    sr = metrics.sr
    T, V = idx.T, idx.V

    if model == "me":
        rhs[V[0]] = 0.0
        rhs[T[-1]] = 0.0
        rhs[V[-1]] = 0.0
        if forced_ep:
            zeta, _ = kern.wall_traction(sr, Req, R, Rd, inv_Ca, inv_Re, alph)
            rhs[V[0]] += (
                2.0 * (n + 2.0) / (n + 1.0) / R
                * (ep_now * (zeta - 3.0 * sr**7 * inv_Re * Rd) + sr**7 * inv_Re * R * epd_now)
            )
    else:
        rhs[T[-1]] = 0.0
        if forced_ep:
            rhs[T[0]] = (
                2.0 * (n + 2.0) * inv_Re * (epd_now * R + ep_now * Rd) / R
                - 2.0 * (n - 1.0) * inv_Re * ep_now * Rd / R
            )
        else:
            rhs[T[0]] = 0.0
    return rhs
