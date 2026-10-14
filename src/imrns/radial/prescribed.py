"""Prescribed radial histories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RadialHistory", "constant_radius", "from_arrays", "sinusoidal_radius"]


@dataclass(frozen=True)
class RadialHistory:
    """A radial history on a time grid, in perturbation-solver scaling.

    ``R``, ``Rd`` and ``Rdd`` are the radius and its first two time derivatives,
    all nondimensional and all the same length as ``t``.
    """

    t: np.ndarray
    R: np.ndarray
    Rd: np.ndarray
    Rdd: np.ndarray

    def __post_init__(self):
        n = self.t.size
        if not (self.R.size == self.Rd.size == self.Rdd.size == n):
            raise ValueError("t, R, Rd and Rdd must all have the same length")
        if n < 2:
            raise ValueError("a history needs at least two time points")
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("t must be strictly increasing")
        if np.any(self.R <= 0):
            raise ValueError("R must be positive")
        for name in ("t", "R", "Rd", "Rdd"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"{name} must be finite")

    def as_kwargs(self) -> dict:
        """Keyword arguments for :func:`imrns.solver.solve_perturbation`."""
        return {"t": self.t, "R": self.R, "Rd": self.Rd, "Rdd": self.Rdd}


def constant_radius(t_end: float, n_steps: int, R: float = 1.0) -> RadialHistory:
    """A bubble held at fixed radius.

    This is what the ``free`` examples use: the perturbation evolves on its own
    with no radial driving.  The operator is then constant in time, so the
    solver caches its factorisation and the run is roughly twenty times faster
    per step than a moving-wall case.
    """
    t = np.linspace(0.0, float(t_end), int(n_steps))
    return RadialHistory(t, np.full_like(t, float(R)), np.zeros_like(t), np.zeros_like(t))


def sinusoidal_radius(t_end: float, n_steps: int, amplitude: float, omega: float,
                      R0: float = 1.0) -> RadialHistory:
    """``R(t) = R0 + a sin(omega t)``, with derivatives taken analytically.

    Useful for exercising the moving-grid terms without needing a radial solver.
    """
    t = np.linspace(0.0, float(t_end), int(n_steps))
    R = R0 + amplitude * np.sin(omega * t)
    if np.any(R <= 0):
        raise ValueError("amplitude is large enough to drive R non-positive")
    return RadialHistory(t, R, amplitude * omega * np.cos(omega * t),
                         -amplitude * omega**2 * np.sin(omega * t))


def from_arrays(t, R, Rd=None, Rdd=None) -> RadialHistory:
    """Wrap arrays from any source, differentiating numerically if needed.

    Supplying ``Rd`` and ``Rdd`` from the radial solver is strongly preferred:
    differencing a saved ``R`` loses accuracy, and ``Rdd`` in particular feeds
    straight into the perturbation forcing.
    """
    t = np.asarray(t, dtype=float).ravel()
    R = np.asarray(R, dtype=float).ravel()
    Rd = np.gradient(R, t) if Rd is None else np.asarray(Rd, dtype=float).ravel()
    Rdd = np.gradient(Rd, t) if Rdd is None else np.asarray(Rdd, dtype=float).ravel()
    return RadialHistory(t, R, Rd, Rdd)
