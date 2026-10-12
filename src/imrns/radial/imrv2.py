"""Bridge to the IMRv2 radial solver.

IMRv2 computes the spherical bubble history including thermal and mass-transfer
physics that this package does not model.  It is a separate project and is not
vendored here.

The two codes use different scalings -- IMRv2 normalises lengths by the maximum
radius, the perturbation solver by the equilibrium radius -- so a history must
be converted before it is handed over.  :func:`adapt_history` does that and is
the only place the conversion happens.

The legacy MATLAB wrapper (``matlab_legacy/common/f_call_IMRv2.m``) searched the
MATLAB path, an environment variable and a sibling directory for ``f_imr_fd``,
and hard-coded that function's option names and output positions.  That coupling
is what this module deliberately avoids: :func:`adapt_history` takes plain
arrays, so any IMRv2 version or any other radial solver can feed it.
"""

from __future__ import annotations

import numpy as np

from ..scales import rescale_radial_history
from .prescribed import RadialHistory

__all__ = ["adapt_history"]


def adapt_history(t, R, Rd, Rdd, *, Rmax: float, Req: float) -> RadialHistory:
    """Convert an IMRv2 history into perturbation-solver scaling.

    Parameters
    ----------
    t, R, Rd, Rdd
        The history as IMRv2 returns it, nondimensionalised by ``Rmax``.
    Rmax, Req
        Maximum and equilibrium radius, in the same units as each other.

    Returns a :class:`~imrns.radial.prescribed.RadialHistory` scaled by ``Req``,
    ready to pass to :func:`imrns.solver.solve_perturbation`.

    Example, with a history already loaded from IMRv2::

        from imrns.radial.imrv2 import adapt_history
        history = adapt_history(t, R, Rd, Rdd, Rmax=150e-6, Req=50e-6)
        result = solve_perturbation(modes=[5, 8], **history.as_kwargs(), ...)
    """
    if Rmax <= 0 or Req <= 0:
        raise ValueError("Rmax and Req must be positive")
    t2, R2, Rd2, Rdd2 = rescale_radial_history(
        np.asarray(t, float), np.asarray(R, float),
        np.asarray(Rd, float), np.asarray(Rdd, float),
        R_from=Rmax, R_to=Req,
    )
    return RadialHistory(t2, R2, Rd2, Rdd2)
