"""JAX assembly: traceable, differentiable, and GPU-capable.

This backend exists for automatic differentiation.  The timestep is

    q^{n+1} = solve(beta0 I - dt A(theta, R, Rd, Rdd),  rhs(q^n, theta)),

which is differentiable end to end, so ``jax.grad`` of any functional of the
trajectory gives exact sensitivities with respect to the material parameters
(``Ca``, ``Re``, ``We``, ``alpha``) and the initial conditions.

Three things make that work, and all three are the reason this is a separate
backend rather than a flag on the NumPy one:

**2F1 is not in JAX.**  The elastic coefficients call the Gauss hypergeometric
function at ``z = 1 - sr^-3``, and ``sr = R/Req`` is differentiable state, so
the derivative is needed, not just the value.  :func:`hyp2f1` wraps SciPy in a
``pure_callback`` and attaches the exact derivative through a ``custom_jvp``
using ``d/dz 2F1(a,b;c;z) = (ab/c) 2F1(a+1,b+1;c+1;z)``.

**Reverse mode would store every operator.**  Keeping ``A`` for all 5000 steps
of a production run is tens of gigabytes.  :func:`scan_trajectory` wraps the
step body in :func:`jax.checkpoint`, so ``A`` is rebuilt from
``(R, Rd, Rdd, theta)`` during the backward pass instead of being stored.

**Numba cannot be mixed in.**  A ``@njit`` kernel cannot be called inside a JAX
trace, so this module reimplements assembly functionally with ``.at[].set()``
in place of in-place writes.  The *physics* is still the shared, unmodified
:mod:`imrns.assemble.kernels`; only the array writes differ.

Scope
-----
Implemented for the ``"me"`` model with either closure and a solved (not
prescribed) amplitude, which covers every bundled example.  Other combinations
raise :class:`NotImplementedError` rather than silently taking a different
code path.
"""

from __future__ import annotations

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    raise ImportError("the jax backend requires jax: pip install 'imrns[jax]'") from exc

from scipy.special import hyp2f1 as _scipy_hyp2f1

from . import kernels as K

jax.config.update("jax_enable_x64", True)

__all__ = ["assemble_lhs", "hyp2f1", "hyp2f1_values", "scan_trajectory", "step_once"]


# --------------------------------------------------------------------------
# Differentiable 2F1
# --------------------------------------------------------------------------


@jax.custom_jvp
def hyp2f1(a, b, c, z):
    """Gauss hypergeometric ``2F1(a, b; c; z)``, differentiable in ``z``.

    Values come from SciPy through a callback; the derivative is supplied
    analytically below.  ``a``, ``b`` and ``c`` depend only on the mode number,
    which is static, so no derivative with respect to them is needed.
    """
    return jax.pure_callback(
        lambda a_, b_, c_, z_: np.asarray(_scipy_hyp2f1(a_, b_, c_, z_), dtype=np.float64),
        jax.ShapeDtypeStruct(jnp.shape(z), jnp.float64),
        a, b, c, z,
        vmap_method="expand_dims",
    )


@hyp2f1.defjvp
def _hyp2f1_jvp(primals, tangents):
    a, b, c, z = primals
    _, _, _, dz = tangents
    return hyp2f1(a, b, c, z), (a * b / c) * hyp2f1(a + 1.0, b + 1.0, c + 1.0, z) * dz


def hyp2f1_values(n, sr):
    """The five ``2F1`` values a mode needs, as a traceable tuple."""
    z = 1.0 - 1.0 / sr**3
    b, c = 4.0 + n / 3.0, 5.0 + n / 3.0
    return (
        hyp2f1(-1.0 / 3.0, b, c, z),
        hyp2f1(2.0 / 3.0, b, c, z),
        hyp2f1(1.0 / 3.0, b, c, z),
        hyp2f1(-2.0 / 3.0, b, c, z),
        hyp2f1(1.0, (17.0 + n) / 3.0, c, z),
    )


# --------------------------------------------------------------------------
# Grid metrics, recomputed inside the trace so they carry derivatives
# --------------------------------------------------------------------------


def _metrics(x, R, Rd, Req, L):
    """Traceable equivalent of :func:`imrns.grid.grid_metrics`."""
    r = R * ((1.0 + x) / (1.0 - x) * L + 1.0)
    drdx = 2.0 * R * L / (1.0 - x) ** 2
    dxdr = (1.0 - x) ** 2 / (2.0 * R * L)
    c1 = -((1.0 - x) ** 3) / (2.0 * R**2 * L**2)
    c2 = (1.0 - x) ** 4 / (4.0 * R**2 * L**2)
    e1 = 3.0 * (1.0 - x) ** 4 / (4.0 * R**3 * L**3)
    e2 = -3.0 * (1.0 - x) ** 5 / (4.0 * R**3 * L**3)
    e3 = (1.0 - x) ** 6 / (8.0 * R**3 * L**3)
    a = Rd / R * (1.0 - x) ** 2 / (2.0 * L) * (1.0 + L * (1.0 + x) / (1.0 - x))
    arg = r**3 - R**3 + Req**3
    rs = jnp.sign(arg) * jnp.abs(arg) ** (1.0 / 3.0)
    vs = Req**3 * ((R / Req) ** 3 - 1.0)
    return dict(r=r, drdx=drdx, dxdr=dxdr, c2=c2, c1=c1, e3=e3, e2=e2, e1=e1,
                a=a, rs=rs, vs=vs, sr=R / Req)


def _upwind(c, Df, Db):
    return jnp.maximum(c, 0.0)[:, None] * Df + jnp.minimum(c, 0.0)[:, None] * Db


SUPPORTED_MODEL = "me"
SUPPORTED_CLOSURES = ("rot", "irr")


def check_supported(model: str = SUPPORTED_MODEL, closure: str = "rot", forced_ep: bool = False):
    """Reject configurations this backend does not implement.

    The JAX path covers the ``"me"`` model with either closure and a solved
    amplitude, which is what every bundled case uses.  The ``"pros"`` model and
    prescribed amplitudes are implemented only in the NumPy and Numba backends.
    Falling through to the ``"me"`` assembly for them would return confident
    wrong gradients, so they are refused here instead.
    """
    if model != SUPPORTED_MODEL:
        raise NotImplementedError(
            f"the jax backend implements only the {SUPPORTED_MODEL!r} model, not {model!r}. "
            "Use backend='numpy' or 'numba' for that model; gradients are not available for it."
        )
    if closure not in SUPPORTED_CLOSURES:
        raise ValueError(f"closure must be one of {SUPPORTED_CLOSURES}, got {closure!r}")
    if forced_ep:
        raise NotImplementedError(
            "the jax backend does not implement prescribed-amplitude (forced_ep) runs. "
            "Use backend='numpy' or 'numba'; gradients are not available for them."
        )


def assemble_lhs(n, xN, closure, beta0, dt, R, Rd, Rdd, Req, L, inv_Ca, inv_Re, inv_We,
                 alpha, x, ops):
    """Build ``beta0 I - dt A`` for one mode, with boundary conditions applied.

    ``ops`` is ``(H1, H2, H3, W, w, One_wT, Df, Db)`` as JAX arrays.  Returns
    ``(lhs, bc_rows)``; ``bc_rows`` are the indices whose right-hand side must
    be zeroed, which :func:`step_once` applies.

    Only the ``"me"`` model is implemented; see :func:`check_supported`.
    """
    check_supported(closure=closure)
    H1, H2, H3, W, w, One_wT, Df, Db = ops
    m = _metrics(x, R, Rd, Req, L)
    r, rs, vs, sr = m["r"], m["rs"], m["vs"], m["sr"]
    dxdr, drdx = m["dxdr"], m["drdx"]
    hyp = hyp2f1_values(n, sr)

    if closure == "irr":
        xi, eta = K.epsilon_coeffs_me(R, Rd, sr, Rdd, n, Req, inv_We, inv_Re, inv_Ca, alpha, hyp)
        A = jnp.array([[0.0, 1.0], [-xi, -eta]])
        return beta0 * jnp.eye(2) - dt * A, ()

    bs = 2 * xN + 2
    T, V, ep, epd = slice(0, xN), slice(xN, 2 * xN), 2 * xN, 2 * xN + 1
    eye = jnp.eye(xN)
    A = jnp.zeros((bs, bs))

    A = A.at[T, T].add(_upwind(m["a"], Df, Db))
    A = A.at[T, V].set(eye)
    A = A.at[ep, epd].set(1.0)

    xi, eta = K.epsilon_coeffs_me(R, Rd, sr, Rdd, n, Req, inv_We, inv_Re, inv_Ca, alpha, hyp)
    A = A.at[epd, ep].add(-xi)
    A = A.at[epd, epd].add(-eta)

    ep_c, epd_c = K.Teq_epsilon_coeffs(R, Rd, Rdd, r, n, inv_Ca, alpha, inv_Re, rs, vs)
    A = A.at[V, ep].add(-ep_c)
    A = A.at[V, epd].add(-epd_c)

    Ct1, Ct2, Ct3, Ct4, Ct5, Ct6, Ct7 = K.non_int_coeffs(r, R, Rd, Rdd, n, inv_Ca, inv_Re, alpha, rs, vs)
    diag = jnp.arange(xN)
    A = A.at[diag + xN, diag].add(-Ct7)
    A = A.at[V, T].add(-((Ct5 * dxdr)[:, None] * H1))
    A = A.at[V, T].add(-((Ct3 * m["c2"])[:, None] * H2))
    A = A.at[V, T].add(-((Ct3 * m["c1"])[:, None] * H1))
    A = A.at[V, T].add(-((Ct1 * m["e3"])[:, None] * H3))
    A = A.at[V, T].add(-((Ct1 * m["e2"])[:, None] * H2))
    A = A.at[V, T].add(-((Ct1 * m["e1"])[:, None] * H1))
    A = A.at[diag + xN, diag + xN].add(-Ct6)
    A = A.at[V, V].add(-((Ct4 * dxdr)[:, None] * H1))
    A = A.at[V, V].add(-((Ct2 * m["c2"])[:, None] * H2))
    A = A.at[V, V].add(-((Ct2 * m["c1"])[:, None] * H1))

    wkap = w * (-(n + 1.0) / (2.0 * n + 1.0) * r ** (-n) * drdx)
    Ckap, Ckapdot = K.kappa_coeffs(r, R, Rd, Rdd, n, inv_Ca, inv_Re, alpha, rs, vs)
    A = A.at[V, T].add(-jnp.outer(Ckap, wkap))
    A = A.at[V, V].add(-jnp.outer(Ckapdot, wkap))

    KmT = (One_wT - W) * (-(r ** (-n)) * drdx)[None, :]
    JmT = W * (r ** (n + 1.0) * drdx)[None, :]
    CJ, CK, CJd, CKd = K.KJ_coeffs(r, rs, vs, n, R, Rd, Rdd, inv_Ca, inv_Re, alpha)
    A = A.at[V, T].add(-(CK[:, None] * KmT + CJ[:, None] * JmT))
    A = A.at[V, V].add(-(CKd[:, None] * KmT + CJd[:, None] * JmT))

    # --- feedback of the rotational field onto the eps'' row -----------------
    quad = w * drdx
    fT_PI, fV_PI, fdT_PI = K.ep_feedback_pointwise(r, R, Rd, n, inv_Ca, inv_Re, alpha, rs)
    A = A.at[epd, T].add(quad * fT_PI + (quad * fdT_PI * dxdr) @ H1)
    A = A.at[epd, V].add(quad * fV_PI)

    # Captured before the moving-grid correction, matching the MATLAB call
    # order; see imrns.assemble.numpy_backend on why this is deliberate.
    Li = A[V, :]

    H_kd, G_kd, F_kap, fT_wall = K.ep_feedback_kappa(R, Rd, Rdd, sr, n, inv_Ca, inv_Re, alpha, hyp)
    A = A.at[epd, :].add(H_kd * (wkap @ Li))
    A = A.at[epd, V].add(G_kd * wkap)
    A = A.at[epd, T].add(F_kap * wkap)
    A = A.at[epd, 0].add(fT_wall)

    fJ_T, fJ_dT, fJ_dt2 = K.ep_feedback_J(r, R, Rd, Rdd, Req, sr, n, inv_Ca, inv_Re, alpha, rs, vs)
    A = A.at[epd, T].add((quad * fJ_T) @ JmT)
    A = A.at[epd, V].add((quad * fJ_dT) @ JmT)
    A = A.at[epd, :].add(((quad * fJ_dt2) @ JmT) @ Li)

    fK_T, fK_dT, fK_dt2 = K.ep_feedback_K(r, R, Rd, Rdd, sr, n, inv_Ca, inv_Re, alpha, rs, vs)
    A = A.at[epd, T].add((quad * fK_T) @ KmT)
    A = A.at[epd, V].add((quad * fK_dT) @ KmT)
    A = A.at[epd, :].add(((quad * fK_dt2) @ KmT) @ Li)

    # --- moving-grid correction (after Li was captured) ----------------------
    A = A.at[V, V].add(_upwind(m["a"], Df, Db))

    lhs = beta0 * jnp.eye(bs) - dt * A

    # --- boundary conditions -------------------------------------------------
    zeta, fT = K.wall_traction(sr, Req, R, Rd, inv_Ca, inv_Re, alpha)
    f_kd, j2k, j2kd = K.wall_kappa_terms(n, R, Rd, sr, zeta, inv_Re)
    v0 = xN

    row = jnp.zeros(bs)
    row = row.at[v0].add(sr**7 / R * inv_Re)
    row = row.at[T].add(sr**7 / R * inv_Re * Rd * dxdr[0] * H1[0, :])
    row = row.at[0].add(fT - j2kd * f_kd)
    row = row.at[T].add(-j2k * wkap)
    row = row.at[V].add(-j2kd * wkap)
    forcing = 2.0 * (n + 2.0) / (n + 1.0) / R
    row = row.at[ep].add(-forcing * (zeta - 3.0 * sr**7 * inv_Re * Rd))
    row = row.at[epd].add(-forcing * (sr**7 * inv_Re * R))
    lhs = lhs.at[v0, :].set(row)

    lhs = lhs.at[xN - 1, :].set(jnp.zeros(bs).at[xN - 1].set(1.0))
    lhs = lhs.at[2 * xN - 1, :].set(jnp.zeros(bs).at[2 * xN - 1].set(1.0))

    return lhs, (v0, xN - 1, 2 * xN - 1)


def step_once(q, n, xN, closure, beta0, rhs, dt, R, Rd, Rdd, Req, L,
              inv_Ca, inv_Re, inv_We, alpha, x, ops):
    """Advance one step: build the operator, apply the BC right-hand side, solve."""
    lhs, bc_rows = assemble_lhs(n, xN, closure, beta0, dt, R, Rd, Rdd, Req, L,
                                inv_Ca, inv_Re, inv_We, alpha, x, ops)
    for row in bc_rows:
        rhs = rhs.at[row].set(0.0)
    return jnp.linalg.solve(lhs, rhs)


def scan_trajectory(q0, n, xN, closure, t, R, Rd, Rdd, Req, L,
                    inv_Ca, inv_Re, inv_We, alpha, x, ops, remat=True):
    """Integrate with backward Euler over the whole history.

    Uses :func:`jax.lax.scan`, with the body wrapped in :func:`jax.checkpoint`
    so reverse-mode differentiation rebuilds each operator instead of storing
    it.  Backward Euler only: BDF2 needs two history levels, which complicates
    the scan carry without changing what the gradients demonstrate.

    Returns ``(q_final, q_history)``.
    """
    dts = jnp.diff(t)

    def body(carry, step):
        dt, Rj, Rdj, Rddj = step
        q_new = step_once(carry, n, xN, closure, 1.0, carry, dt, Rj, Rdj, Rddj,
                          Req, L, inv_Ca, inv_Re, inv_We, alpha, x, ops)
        return q_new, q_new

    if remat:
        body = jax.checkpoint(body)
    return jax.lax.scan(body, q0, (dts, R[1:], Rd[1:], Rdd[1:]))


def make_ops(xN, h):
    """Build the operator matrices this backend needs, as JAX arrays."""
    from ..operators import _backward_d1, _forward_d1, difference_stencil, trapezoid_matrices
    from ..grid import computational_grid

    H1, H2, H3 = difference_stencil(xN, h)
    W, w, One_wT = trapezoid_matrices(computational_grid(xN))
    return tuple(
        jnp.asarray(a)
        for a in (H1, H2, H3, W, w, One_wT, _forward_d1(xN, float(h)), _backward_d1(xN, float(h)))
    )
