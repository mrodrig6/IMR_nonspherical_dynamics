"""Automatic differentiation through the solver, checked against finite differences.

These are the tests that justify having a JAX backend at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from imrns.backends import available_backends

pytestmark = pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")


def _setup(n_steps=60):
    t = np.linspace(0.0, 1.0, n_steps)
    return dict(t=t, R=1 + 0.1 * np.sin(2 * t), Rd=0.2 * np.cos(2 * t),
                Rdd=-0.4 * np.sin(2 * t), mode=5, xN=24, ep0=0.01)


def _loss_fn(closure):
    import jax.numpy as jnp

    from imrns.sensitivity import amplitude_trajectory

    cfg = _setup()

    def loss(params):
        ep = amplitude_trajectory(params, closure=closure, **cfg)
        return jnp.sum(ep**2)

    return loss


@pytest.mark.parametrize("closure", ["irr", "rot"])
@pytest.mark.parametrize("name", ["Ca", "Re", "We", "alpha"])
def test_gradient_matches_central_difference(closure, name):
    import jax

    from imrns.sensitivity import default_parameters

    loss = jax.jit(_loss_fn(closure))
    params = default_parameters()
    grad = jax.grad(loss)(params)

    step = 1e-5 * max(abs(params[name]), 1.0)
    hi = dict(params, **{name: params[name] + step})
    lo = dict(params, **{name: params[name] - step})
    fd = (loss(hi) - loss(lo)) / (2 * step)

    scale = max(abs(float(fd)), abs(float(grad[name])), 1e-30)
    assert abs(float(grad[name]) - float(fd)) / scale < 2e-5, (
        f"{closure}/{name}: AD {float(grad[name]):.6e} vs FD {float(fd):.6e}"
    )


def test_gradient_flows_through_hypergeometric_terms():
    """The elastic coefficients call 2F1 at an argument that depends on R.

    If the custom_jvp were missing or wrong, the alpha sensitivity -- which
    reaches the solution only through those terms -- would be zero or wrong.
    """
    import jax

    from imrns.sensitivity import default_parameters

    grad = jax.grad(_loss_fn("irr"))(default_parameters())
    assert abs(float(grad["alpha"])) > 0.0


def test_checkpointing_does_not_change_the_gradient():
    """Rematerialising the operator in the backward pass is exact, not approximate."""
    import jax
    import jax.numpy as jnp

    from imrns.sensitivity import amplitude_trajectory, default_parameters

    cfg = _setup(n_steps=40)

    def make(remat):
        def loss(params):
            return jnp.sum(amplitude_trajectory(params, closure="rot", remat=remat, **cfg) ** 2)

        return loss

    params = default_parameters()
    with_remat = jax.grad(make(True))(params)
    without = jax.grad(make(False))(params)
    for key in params:
        assert float(with_remat[key]) == pytest.approx(float(without[key]), rel=1e-10)


def test_jax_trajectory_matches_the_reference_solver():
    """The differentiable path solves the same problem as the production solver."""
    import numpy as np

    from imrns.sensitivity import amplitude_trajectory, default_parameters
    from imrns.solver import solve_perturbation

    cfg = _setup(n_steps=80)
    params = default_parameters()
    ep_jax = np.asarray(amplitude_trajectory(params, closure="rot", **cfg))
    ref = solve_perturbation(
        modes=cfg["mode"], t=cfg["t"], R=cfg["R"], Rd=cfg["Rd"], Rdd=cfg["Rdd"],
        Ca=params["Ca"], Re=params["Re"], We=params["We"], alpha=params["alpha"],
        xN=cfg["xN"], ep0=cfg["ep0"], closure="rot",
        method=1,  # the differentiable path uses backward Euler throughout
    )
    np.testing.assert_allclose(ep_jax, ref.ep[0], rtol=1e-9, atol=1e-14)
