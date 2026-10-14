"""Special functions.

The Gauss hypergeometric ``2F1`` appears in the closed-form elastic
coefficients.  The MATLAB code shipped a compiled MEX binary for it (two
platform builds plus seven Cephes C files).  SciPy provides the same function,
so the compiled dependency is gone, along with the ``mex -setup`` requirement
and the platform-specific binaries.

``2F1`` is evaluated at most five times per mode per timestep, always at scalar
arguments, so there is no performance reason to do anything cleverer.
"""

from __future__ import annotations

import numpy as np
from scipy.special import hyp2f1 as _scipy_hyp2f1

__all__ = ["hyp2f1", "hyp2f1_values"]


def hyp2f1(a, b, c, z):
    """Gauss hypergeometric ``2F1(a, b; c; z)`` for real arguments."""
    return _scipy_hyp2f1(a, b, c, z)


def hyp2f1_values(n: float, sr: float) -> tuple[float, ...]:
    """The five ``2F1`` values a mode needs at wall stretch ``sr``.

    Order matches :func:`imrns.assemble.kernels.hyp_arguments`.
    """
    from .assemble.kernels import hyp_arguments

    return tuple(float(_scipy_hyp2f1(a, b, c, z)) for a, b, c, z in hyp_arguments(n, sr))


def d_hyp2f1_dz(a, b, c, z):
    """``d/dz 2F1(a, b; c; z) = (a b / c) 2F1(a+1, b+1; c+1; z)``.

    Used by the JAX backend to differentiate through the elastic coefficients,
    where the argument ``z = 1 - sr^-3`` depends on the bubble radius.
    """
    return (np.asarray(a) * np.asarray(b) / np.asarray(c)) * _scipy_hyp2f1(
        np.asarray(a) + 1.0, np.asarray(b) + 1.0, np.asarray(c) + 1.0, z
    )
