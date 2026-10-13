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

# The midpoint, #4E3629 -- the house colour the original figures centre on.
_BROWN = np.array([0x4E, 0x36, 0x29]) / 255.0

# Hue anchors for the two arms, from the midpoint outwards.  Luminance is
# imposed afterwards, so these set hue and saturation only.
_COOL_ARM = [_BROWN, np.array([0.13, 0.40, 0.67]), np.array([0.72, 0.85, 0.96])]
_WARM_ARM = [_BROWN, np.array([0.85, 0.62, 0.18]), np.array([0.98, 0.93, 0.78])]

_SRGB_TO_Y = np.array([0.2126, 0.7152, 0.0722])


def _to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def relative_luminance(rgb):
    """Relative luminance of sRGB values in [0, 1]."""
    return _to_linear(np.asarray(rgb, dtype=float)) @ _SRGB_TO_Y


def _interpolate(stops, n):
    """Piecewise-linear interpolation through evenly spaced RGB stops."""
    stops = np.asarray(stops, dtype=float)
    positions = np.linspace(0.0, 1.0, len(stops))
    query = np.linspace(0.0, 1.0, int(n))
    return np.clip(
        np.column_stack([np.interp(query, positions, stops[:, k]) for k in range(3)]), 0.0, 1.0
    )


def _force_luminance(rgb, target):
    """Rescale linear RGB to hit ``target`` luminance, keeping chromaticity.

    Where the target cannot be reached without clipping (a saturated hue has a
    bounded luminance), the colour is mixed towards white instead, which raises
    luminance while desaturating rather than distorting the hue.
    """
    out = np.empty_like(rgb)
    for i, (c, t) in enumerate(zip(rgb, np.atleast_1d(target))):
        lin = _to_linear(c)
        y = lin @ _SRGB_TO_Y
        if y <= 1e-9:
            out[i] = _to_srgb(np.full(3, t))
            continue
        scaled = lin * (t / y)
        if scaled.max() > 1.0:
            # Cannot scale further; blend towards white to make up the shortfall.
            lin_n = lin / max(lin.max(), 1e-12)
            y_n = lin_n @ _SRGB_TO_Y
            if t <= y_n:
                scaled = lin_n * (t / max(y_n, 1e-12))
            else:
                w = (t - y_n) / max(1.0 - y_n, 1e-12)
                scaled = (1 - w) * lin_n + w
        out[i] = _to_srgb(scaled)
    return out


def brown_centered_diverging(n: int = 256, gamma: float = 0.85) -> np.ndarray:
    """Diverging map with the house brown midpoint, symmetric in luminance.

    Cool blue for negative values, warm gold for positive, through ``#4E3629``
    at zero.  Built for signed fields such as strain, where the midpoint marks
    zero.

    The two arms are given an **identical luminance profile**, so a value of
    ``-x`` and one of ``+x`` read as equally intense and the sign is carried by
    hue alone.  Getting that wrong is easy and consequential: an earlier version
    of this ramp differed by a factor of 3.3 in luminance at the half-way point,
    which made compressive strain look far stronger than tensile strain of the
    same magnitude.

    ``gamma`` shapes how fast luminance rises away from the midpoint; the
    default spreads contrast fairly evenly over the range.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    half = int(np.ceil(n / 2))
    s = np.linspace(0.0, 1.0, half)

    y_mid = float(relative_luminance(_BROWN))
    y_end = 0.86
    target = y_mid + (y_end - y_mid) * s**gamma

    cool = _force_luminance(_interpolate(_COOL_ARM, half), target)
    warm = _force_luminance(_interpolate(_WARM_ARM, half), target)

    full = np.vstack([cool[::-1], warm[1:]]) if n % 2 else np.vstack([cool[::-1], warm])
    # Resample to exactly n rows.
    idx = np.linspace(0.0, len(full) - 1, n)
    lo, hi = np.floor(idx).astype(int), np.ceil(idx).astype(int)
    frac = (idx - lo)[:, None]
    return np.clip(full[lo] * (1 - frac) + full[hi] * frac, 0.0, 1.0)


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
