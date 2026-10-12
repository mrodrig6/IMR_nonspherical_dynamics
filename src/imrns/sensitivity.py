"""Parameter sensitivities by automatic differentiation.

The whole timestepper is differentiable, so gradients of any functional of the
trajectory with respect to material parameters or initial conditions are exact
-- no finite-difference step size to choose, and one reverse-mode pass gives
every parameter at once.

Typical use, fitting a measured amplitude history:

    from imrns.sensitivity import amplitude_trajectory
    import jax

    def loss(params):
        ep = amplitude_trajectory(params, t=t, R=R, Rd=Rd, Rdd=Rdd, mode=5, xN=64)
        return jnp.sum((ep - measured) ** 2)

    grad = jax.grad(loss)({"Ca": 50.0, "Re": 35.0, "We": 45.0, "alpha": 0.1})

Parameters are passed as a dict so ``jax.grad`` returns a dict of the same
shape.  ``Ca``, ``Re`` and ``We`` enter as reciprocals internally, which keeps
the inviscid and inelastic limits finite and differentiable.

Backward Euler only.  BDF2 would need two history levels in the scan carry
without changing anything about the sensitivities themselves; use
:func:`imrns.solver.solve_perturbation` for production trajectories and this
module for gradients.
"""

from __future__ import annotations

try:
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sensitivity analysis requires jax: pip install 'imrns[jax]'"
    ) from exc

from .assemble import jax_backend as jb
from .grid import computational_grid, grid_spacing

__all__ = ["amplitude_trajectory", "default_parameters", "field_trajectory"]


def default_parameters(Ca=50.0, Re=35.0, We=45.0, alpha=0.1) -> dict:
    """A parameter dict in the form the trajectory functions expect."""
    return {"Ca": float(Ca), "Re": float(Re), "We": float(We), "alpha": float(alpha)}


def _trajectory(params, *, t, R, Rd, Rdd, mode, xN, L=5.0, Req=1.0,
                ep0=0.0, epd0=0.0, closure="rot", remat=True):
    x_np = computational_grid(xN)
    h = grid_spacing(x_np)
    x = jnp.asarray(x_np)
    ops = jb.make_ops(xN, h)

    inv_Ca = 1.0 / params["Ca"]
    inv_Re = 1.0 / params["Re"]
    inv_We = 1.0 / params["We"]
    alpha = params["alpha"]

    if closure == "irr":
        q0 = jnp.array([ep0, epd0])
    else:
        q0 = jnp.zeros(2 * xN + 2).at[2 * xN].set(ep0).at[2 * xN + 1].set(epd0)

    _, history = jb.scan_trajectory(
        q0, float(mode), xN, closure,
        jnp.asarray(t), jnp.asarray(R), jnp.asarray(Rd), jnp.asarray(Rdd),
        Req, L, inv_Ca, inv_Re, inv_We, alpha, x, ops, remat=remat,
    )
    # Prepend the initial state so the history lines up with t.
    return jnp.concatenate([q0[None, :], history], axis=0)


def amplitude_trajectory(params, *, t, R, Rd, Rdd, mode, xN, **kwargs):
    """Differentiable amplitude history ``eps(t)`` for one mode.

    Returns an array of length ``len(t)``.
    """
    closure = kwargs.get("closure", "rot")
    q = _trajectory(params, t=t, R=R, Rd=Rd, Rdd=Rdd, mode=mode, xN=xN, **kwargs)
    return q[:, 0] if closure == "irr" else q[:, 2 * xN]


def field_trajectory(params, *, t, R, Rd, Rdd, mode, xN, **kwargs):
    """Differentiable rotational field history ``T(r, t)``, shape ``(len(t), xN)``."""
    if kwargs.get("closure", "rot") == "irr":
        raise ValueError('the irrotational closure has no rotational field; use closure="rot"')
    q = _trajectory(params, t=t, R=R, Rd=Rd, Rdd=Rdd, mode=mode, xN=xN, **kwargs)
    return q[:, :xN]
