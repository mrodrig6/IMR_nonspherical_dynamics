"""Golden-trajectory regression.

These pin the current numerics so a later refactor cannot change answers
silently.  They say nothing about whether the answers are *right* -- that is
what :mod:`test_kernels_vs_matlab` and :mod:`test_physics_limits` are for.

Regenerate deliberately, and only when a change is intended and understood:

    python tests/make_golden.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imrns.solver import solve_perturbation

GOLDEN = Path(__file__).parent / "data" / "golden_trajectories.npz"

CASES = {
    "free_me_rot": dict(modes=[2, 5, 8], ep0=[0.1, -0.1, 0.05], n_steps=300, t_end=3.0,
                        drive="free", model="me", closure="rot", method=2),
    "free_me_irr": dict(modes=[5], ep0=[0.1], n_steps=300, t_end=3.0,
                        drive="free", model="me", closure="irr", method=2),
    "free_pros_rot": dict(modes=[5], ep0=[0.1], n_steps=300, t_end=3.0,
                          drive="free", model="pros", closure="rot", method=2),
    "osc_me_rot": dict(modes=[5, 8], ep0=[0.05, -0.03], n_steps=250, t_end=2.0,
                       drive="osc", model="me", closure="rot", method=2),
    "osc_me_rot_be": dict(modes=[5], ep0=[0.05], n_steps=250, t_end=2.0,
                          drive="osc", model="me", closure="rot", method=1),
}


def run(spec):
    t = np.linspace(0.0, spec["t_end"], spec["n_steps"])
    if spec["drive"] == "osc":
        R, Rd, Rdd = 1 + 0.15 * np.sin(3 * t), 0.45 * np.cos(3 * t), -1.35 * np.sin(3 * t)
    else:
        R, Rd, Rdd = np.ones_like(t), np.zeros_like(t), np.zeros_like(t)
    return solve_perturbation(
        modes=spec["modes"], ep0=spec["ep0"], t=t, R=R, Rd=Rd, Rdd=Rdd,
        Ca=50.0, Re=35.0, We=45.0, alpha=0.1, xN=48,
        model=spec["model"], closure=spec["closure"], method=spec["method"],
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_trajectory_matches_golden(name):
    golden = np.load(GOLDEN)
    result = run(CASES[name])
    np.testing.assert_allclose(result.ep, golden[f"{name}__ep"], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(result.T, golden[f"{name}__T"], rtol=1e-10, atol=1e-14)


def test_golden_fixture_is_present():
    """A missing fixture must fail, not skip.

    These are committed, so absence means something went wrong with the
    checkout or the ignore rules -- silently skipping would hide a regression
    suite that is not running at all.
    """
    assert GOLDEN.exists(), f"{GOLDEN} is missing; regenerate with tests/make_golden.py"


def test_golden_covers_every_model_and_closure():
    """The fixtures must exercise both models, both closures and both time schemes."""
    specs = list(CASES.values())
    assert {s["model"] for s in specs} == {"me", "pros"}
    assert {s["closure"] for s in specs} == {"rot", "irr"}
    assert {s["method"] for s in specs} == {1, 2}
    assert {s["drive"] for s in specs} == {"free", "osc"}
