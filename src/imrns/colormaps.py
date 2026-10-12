"""Colormaps used by the figures.

The MATLAB repository referenced a ``cmap/`` directory from ``setup_paths``, the
README and every example configuration, but the directory was never committed.
Any example with snapshot output therefore failed on an undefined
``brown_centered_diverging``, and the solver's quasi-equilibrium branch called
``magma``, which is not a MATLAB builtin either.  Both are defined here.

Each function returns an ``(n, 3)`` array of RGB values in ``[0, 1]``, which is
accepted directly by matplotlib.
"""

from __future__ import annotations

import numpy as np

__all__ = ["available", "brown_centered_diverging", "get"]

# The neutral midpoint, #4E3629.
_BROWN = np.array([0x4E, 0x36, 0x29]) / 255.0
_COOL = np.array([0.13, 0.40, 0.67])
_WARM = np.array([0.85, 0.65, 0.20])


def _interpolate(stops, n):
    """Piecewise-linear interpolation through evenly spaced RGB stops."""
    stops = np.asarray(stops, dtype=float)
    positions = np.linspace(0.0, 1.0, len(stops))
    query = np.linspace(0.0, 1.0, int(n))
    return np.clip(
        np.column_stack([np.interp(query, positions, stops[:, k]) for k in range(3)]), 0.0, 1.0
    )


def brown_centered_diverging(n: int = 256) -> np.ndarray:
    """Diverging map with a dark brown neutral midpoint.

    Built for signed strain fields, where the midpoint marks zero.  Cool blue
    for negative values, warm gold for positive, through ``#4E3629`` at the
    centre.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    return _interpolate(
        [
            np.array([0.85, 0.92, 0.98]),
            _COOL,
            _BROWN,
            _WARM,
            np.array([0.99, 0.95, 0.80]),
        ],
        n,
    )


_REGISTRY = {"brown_centered_diverging": brown_centered_diverging}


def available() -> list[str]:
    """Names of the colormaps defined here."""
    return sorted(_REGISTRY)


def get(name, n: int = 256) -> np.ndarray:
    """Look up a colormap by name, or pass through a callable or array.

    Falls back to matplotlib's registry for any name not defined here, so
    ``get("magma")`` works without this module having to reimplement it.
    """
    if isinstance(name, np.ndarray):
        return name
    if callable(name):
        return np.asarray(name(n))
    key = str(name)
    if key in _REGISTRY:
        return _REGISTRY[key](n)
    try:
        import matplotlib

        return matplotlib.colormaps[key](np.linspace(0, 1, n))[:, :3]
    except Exception as exc:
        raise ValueError(
            f"unknown colormap {key!r}; this package defines {available()}"
        ) from exc
