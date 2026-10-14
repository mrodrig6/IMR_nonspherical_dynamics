"""Top-level nonspherical perturbation solver.

Replaces ``matlab_legacy/common/compute_rotational_perturbation_evolution.m``.

What it computes
----------------
Given a radial history ``R(t)`` of a spherical bubble, the linear evolution of
small nonspherical perturbations of the interface,

    r_wall(theta, t) = R(t) + sum_n eps_n(t) Y_n(theta),

together with the rotational (toroidal) field ``T_n(r, t)`` each mode drives in
the surrounding medium.  Two closures are available: ``"rot"`` solves the
coupled amplitude-plus-field system, ``"irr"`` drops the field and keeps only
the potential-flow amplitude equation.

Differences from the MATLAB original
------------------------------------
* The modes are solved as independent blocks rather than one large system; see
  :mod:`imrns.assemble.numpy_backend`.
* The solver no longer draws figures.  Early termination is reported through
  :attr:`Result.termination` and the returned arrays are truncated
  consistently -- the legacy quasi-equilibrium branch truncated the field and
  amplitude histories but returned ``t``, ``R`` and ``Rd`` at full length.
* All operators use the single grid spacing from
  :func:`imrns.grid.grid_spacing`.  The original built the derivative stencils
  with ``(2 - 1e-4)/(xN - 1)`` and the advection operator with ``2/(xN - 1)``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from . import grid as gridmod
from .assemble import numpy_backend as npb
from .assemble import kernels as K
from .backends import get_backend
from .layout import build_layout
from .operators import difference_stencil, trapezoid_matrices, upwind_advection
from .stepper import BlockSolver, bdf_coefficients, is_operator_constant, row_equilibrate

__all__ = ["PerturbationResult", "solve_perturbation"]


@dataclass
class PerturbationResult:
    """Output of :func:`solve_perturbation`.

    All arrays are trimmed to the same number of time points, so
    ``len(t) == ep.shape[1]`` always holds.
    """

    t: np.ndarray
    R: np.ndarray
    Rd: np.ndarray
    ep: np.ndarray
    epd: np.ndarray
    T: np.ndarray
    Td: np.ndarray
    modes: tuple[int, ...]
    termination: str = "completed"
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    def mode(self, n: int) -> int:
        """Row index of mode number ``n``."""
        return self.modes.index(n)


def _as_mode_vector(values, n_modes, name):
    values = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
    if values.size == 1 and n_modes > 1:
        values = np.repeat(values, n_modes)
    if values.size != n_modes:
        raise ValueError(f"{name} must be scalar or have one value per mode, got {values.size}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    return values


def _as_history(values, name, n_time=None):
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    if n_time is not None and values.size != n_time:
        raise ValueError(f"{name} must have {n_time} entries to match t, got {values.size}")
    return values


def _as_forced(history, n_modes, n_time, name):
    history = np.asarray(history, dtype=float)
    if history.ndim == 1 and n_modes == 1:
        history = history.reshape(1, -1)
    elif history.shape == (n_time, n_modes):
        history = history.T
    if history.shape != (n_modes, n_time):
        raise ValueError(f"{name} must have shape ({n_modes}, {n_time}), got {history.shape}")
    if not np.all(np.isfinite(history)):
        raise ValueError(f"{name} must be finite")
    return history


def solve_perturbation(
    *,
    modes,
    t,
    R,
    Rd,
    Rdd,
    Ca,
    Re,
    We,
    alpha=0.0,
    Req=1.0,
    xN=256,
    L=5.0,
    ep0=0.0,
    epd0=0.0,
    T0=None,
    Td0=None,
    model="me",
    closure="rot",
    method=2,
    forced_ep=None,
    forced_epd=None,
    backend="numpy",
    reuse_factorisation=None,
    progress=False,
):
    """Evolve the nonspherical perturbation over the supplied radial history.

    Parameters
    ----------
    modes
        Spherical-harmonic mode number, or a sequence of them.
    t, R, Rd, Rdd
        Nondimensional time grid and the radial history on it.  All four must
        have the same length.  The solver does not synthesise a radial history:
        for a free constant-radius run pass ``R = ones_like(t)`` and zeros.
    Ca, Re, We
        Cauchy, Reynolds and Weber numbers from :func:`imrns.scales.compute_scales`.
        Infinite values are accepted and mean the corresponding stress vanishes.
    alpha
        Second-order coefficient of the quadratic strain-energy law.
    Req
        Equilibrium radius in solver units, normally ``1``.
    xN, L
        Number of radial grid points and the stretching parameter of the
        coordinate map.
    ep0, epd0
        Initial amplitude and its time derivative, one value per mode.
    T0, Td0
        Initial rotational field and its time derivative, shape
        ``(n_modes, xN)``.  Default zero.
    model
        ``"me"`` or ``"pros"``.
    closure
        ``"rot"`` or ``"irr"``.
    method
        ``1`` for backward Euler throughout, ``2`` for BDF2 after the first step.
    forced_ep, forced_epd
        Prescribed amplitude history of shape ``(n_modes, len(t))``.  Supplying
        them switches the amplitude out of the state vector.
    backend
        ``"numpy"``, ``"numba"`` or ``"jax"``.
    reuse_factorisation
        Force factorisation caching on or off.  The default detects a constant
        operator automatically.
    progress
        Print a percentage counter to stderr.

    Returns a :class:`PerturbationResult`.
    """
    backend_impl = get_backend(backend)

    t = _as_history(t, "t")
    n_time = t.size
    R = _as_history(R, "R", n_time)
    Rd = _as_history(Rd, "Rd", n_time)
    Rdd = _as_history(Rdd, "Rdd", n_time)
    if n_time < 2:
        raise ValueError("t must contain at least two points")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")
    if np.any(R <= 0):
        raise ValueError("R must be positive")

    is_forced = forced_ep is not None or forced_epd is not None
    layout = build_layout(modes, xN, model, closure, is_forced)
    n_modes = layout.n_modes

    ep0 = _as_mode_vector(ep0, n_modes, "ep0")
    epd0 = _as_mode_vector(epd0, n_modes, "epd0")

    if is_forced:
        if forced_ep is None or forced_epd is None:
            raise ValueError("forced_ep and forced_epd must be supplied together")
        ep_hist = _as_forced(forced_ep, n_modes, n_time, "forced_ep")
        epd_hist = _as_forced(forced_epd, n_modes, n_time, "forced_epd")
    else:
        ep_hist = np.zeros((n_modes, n_time))
        epd_hist = np.zeros((n_modes, n_time))
        ep_hist[:, 0] = ep0
        epd_hist[:, 0] = epd0

    T_hist = np.zeros((n_modes, xN, n_time))
    Td_hist = np.zeros((n_modes, xN, n_time))
    if T0 is not None:
        T_hist[:, :, 0] = np.asarray(T0, dtype=float).reshape(n_modes, xN)
    if Td0 is not None:
        Td_hist[:, :, 0] = np.asarray(Td0, dtype=float).reshape(n_modes, xN)

    inv_Ca = 0.0 if not np.isfinite(Ca) else 1.0 / Ca
    inv_Re = 0.0 if not np.isfinite(Re) else 1.0 / Re
    inv_We = 0.0 if not np.isfinite(We) else 1.0 / We
    groups = (inv_Ca, inv_Re, inv_We)

    if closure == "rot" and inv_Ca == 0.0 and inv_Re == 0.0:
        # The wall condition is continuity of tangential traction.  With neither
        # elasticity nor viscosity there is no deviatoric stress, every term in
        # that row vanishes, and the block becomes singular.  The rotational
        # field is also physically absent in this limit.
        raise ValueError(
            "the rotational closure is undefined when both Ca and Re are infinite: "
            "with no deviatoric stress the tangential-traction wall condition is "
            'identically zero. Use closure="irr", or give a finite Ca or Re.'
        )

    x = gridmod.computational_grid(xN)
    h = gridmod.grid_spacing(x)
    H1, H2, H3 = difference_stencil(xN, h)
    W, w, One_wT = trapezoid_matrices(x)
    ops = (H1, H2, H3, W, w, One_wT)

    # Initial state, laid out block by block.
    q = np.zeros(layout.system_size)
    for i in range(n_modes):
        idx = layout.per_mode[i]
        if layout.has_field:
            q[idx.T] = T_hist[i, :, 0]
            if idx.V is not None:
                q[idx.V] = Td_hist[i, :, 0]
        if idx.ep is not None:
            q[idx.ep] = ep_hist[i, 0]
            q[idx.epd] = epd_hist[i, 0]

    eps_rows_local = []
    for i in range(n_modes):
        loc = layout.local(i)
        eps_rows_local.append(
            np.array([], dtype=int) if loc.ep is None else np.array([loc.ep, loc.epd])
        )

    if reuse_factorisation is None:
        reuse_factorisation = is_operator_constant(t, R, Rd, Rdd)
    solver = BlockSolver(reuse=bool(reuse_factorisation))

    has_forced_source = is_forced and model == "me" and closure == "rot"

    q_prev = None
    bs = layout.block_size
    next_pct = 1
    termination = "completed"

    for i in range(n_time - 1):
        j = i + 1  # the operator is evaluated at the new time level
        dt_n = t[j] - t[i]
        dt_prev = None if i == 0 else t[i] - t[i - 1]
        coef = bdf_coefficients(dt_n, dt_prev, method)
        key = "be" if coef.is_backward_euler else "bdf2"

        metrics = gridmod.grid_metrics(x, R[j], Rd[j], Req, L)
        state = (R[j], Rd[j], Rdd[j], Req)
        # The advection operator is mode independent; build it once per step.
        aadv = upwind_advection(metrics.a, metrics.h) if layout.has_field else None

        rhs = q.copy() if coef.is_backward_euler else coef.beta1 * q + coef.beta2 * q_prev

        if has_forced_source:
            for m in range(n_modes):
                idx = layout.per_mode[m]
                rhs[idx.V] += dt_n * K.forced_source(
                    ep_hist[m, j], epd_hist[m, j], R[j], Rd[j], Rdd[j], metrics.r,
                    float(layout.modes[m]), inv_Ca, inv_Re, alpha, metrics.rs, metrics.vs,
                )

        cached = solver.get(key)
        blocks, rhs_blocks, scales = [], [], []

        for m in range(n_modes):
            loc = layout.local(m)
            n = layout.modes[m]
            rhs_m = rhs[layout.per_mode[m].block].copy()
            ep_now, epd_now = ep_hist[m, j], epd_hist[m, j]

            if cached is not None:
                if layout.has_field:
                    npb.apply_boundary_rhs(
                        rhs_m, n, loc, metrics, state, groups, alpha, model,
                        layout.forced_ep, ep_now, epd_now,
                    )
                scale = cached.scales[m]
                rhs_m /= scale
            else:
                hyp = backend_impl.hyp2f1_values(float(n), metrics.sr)
                A = backend_impl.assemble_block(
                    n, loc, metrics, state, groups, alpha, model, layout.closure,
                    layout.forced_ep, ops, hyp, aadv=aadv,
                )
                if not layout.forced_ep and layout.closure != "irr":
                    backend_impl.ep_feedback_rows(
                        A, n, loc, metrics, state, groups, alpha, model, ops, hyp
                    )
                if layout.has_field:
                    backend_impl.add_moving_grid_terms(A, loc, metrics, model, aadv=aadv)

                lhs_m = -dt_n * A
                lhs_m[np.diag_indices(bs)] += coef.beta0
                if layout.has_field:
                    npb.apply_boundary_conditions(
                        lhs_m, rhs_m, n, loc, metrics, state, groups, alpha,
                        model, layout.forced_ep, ops, ep_now, epd_now,
                    )
                scales.append(row_equilibrate(lhs_m, rhs_m, eps_rows_local[m]))
                blocks.append(lhs_m)

            rhs_blocks.append(rhs_m)

        if cached is None:
            cached = solver.store(key, blocks, scales)

        q_new = np.concatenate(BlockSolver.solve(cached, rhs_blocks))

        if not np.all(np.isfinite(q_new)):
            termination = f"non-finite solution at step {j}"
            warnings.warn(
                f"solution became non-finite at step {j}; results truncated there",
                RuntimeWarning, stacklevel=2,
            )
            T_hist, Td_hist = T_hist[:, :, :j], Td_hist[:, :, :j]
            ep_hist, epd_hist = ep_hist[:, :j], epd_hist[:, :j]
            t, R, Rd = t[:j], R[:j], Rd[:j]
            break

        for m in range(n_modes):
            idx = layout.per_mode[m]
            if layout.has_field:
                T_hist[m, :, j] = q_new[idx.T]
                if idx.V is not None:
                    Td_hist[m, :, j] = q_new[idx.V]
            if idx.ep is not None:
                ep_hist[m, j] = q_new[idx.ep]
                epd_hist[m, j] = q_new[idx.epd]

        q_prev, q = q, q_new

        if progress:
            pct = int(100 * (i + 1) / (n_time - 1))
            if pct >= next_pct:
                print(f"\r{pct:3d}% complete", end="", flush=True)
                next_pct = pct + 1

    if progress:
        print()

    return PerturbationResult(
        t=t, R=R, Rd=Rd,
        ep=ep_hist, epd=epd_hist, T=T_hist, Td=Td_hist,
        modes=layout.modes,
        termination=termination,
        diagnostics={
            "backend": backend,
            "reused_factorisation": bool(reuse_factorisation),
            "block_size": layout.block_size,
            "system_size": layout.system_size,
        },
    )
