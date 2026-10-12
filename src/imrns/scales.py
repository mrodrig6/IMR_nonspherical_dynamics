"""Nondimensionalisation.

Every dimensionless group the solver uses is defined here, once.  In the legacy
code the same five lines were repeated in ``s_basic_simulation.m`` and
``run_nonspherical_example.m``, with the radial history rescaled afterwards by
hand in both places.

The reference scales are the equilibrium radius, the ambient density and the
ambient pressure:

    Lc = Req,  tc = Lc sqrt(rho / p_inf),  Uc = Lc / tc,  pc = rho Uc^2

Note ``pc = rho (Lc/tc)^2 = p_inf`` identically, so the pressure scale is just
the ambient pressure.

Resulting groups:

    Ca = pc / G                      Cauchy number, inertia over elasticity
    Re = Lc sqrt(rho pc) / mu        Reynolds number, inertia over viscosity
    We = pc Lc / (2 sigma)           Weber number, inertia over surface tension
    Oh = sqrt(We) / Re               Ohnesorge number
    De = Ca / Re                     Deborah number
    Ec = sqrt(We) / Ca               elastocapillary number

A vanishing material property gives an infinite group, and the solver only ever
uses reciprocals (``1/Ca``, ``1/Re``, ``1/We``), so the limits are well behaved:
``G = 0`` means ``Ca = inf`` and no elastic stress.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scales:
    """Characteristic scales and the dimensionless groups built from them."""

    Lc: float
    tc: float
    Uc: float
    pc: float
    Ca: float
    Re: float
    We: float
    Oh: float
    De: float
    Ec: float

    def summary(self) -> str:
        return (
            f"Ca={self.Ca:.4g}  Re={self.Re:.4g}  We={self.We:.4g}  "
            f"Oh={self.Oh:.4g}  De={self.De:.4g}  Ec={self.Ec:.4g}"
        )


def _reciprocal_safe(numerator: float, denominator: float) -> float:
    """``numerator/denominator``, returning ``inf`` for a zero denominator."""
    if denominator == 0.0:
        return np.inf
    return numerator / denominator


def compute_scales(Req: float, rho: float, p_inf: float, mu: float, G: float, sigma: float) -> Scales:
    """Build the dimensionless groups from dimensional material properties.

    Parameters are SI: ``Req`` in m, ``rho`` in kg/m^3, ``p_inf`` in Pa,
    ``mu`` in Pa s, ``G`` in Pa, ``sigma`` in N/m.
    """
    if Req <= 0:
        raise ValueError(f"Req must be positive, got {Req}")
    if rho <= 0 or p_inf <= 0:
        raise ValueError("rho and p_inf must be positive")
    if mu < 0 or G < 0 or sigma < 0:
        raise ValueError("mu, G and sigma must be non-negative")

    Lc = Req
    tc = Lc * np.sqrt(rho / p_inf)
    Uc = Lc / tc
    pc = rho * Uc**2

    Ca = _reciprocal_safe(pc, G)
    Re = _reciprocal_safe(Lc * np.sqrt(rho * pc), mu)
    We = _reciprocal_safe(pc * Lc, 2.0 * sigma)
    Oh = np.sqrt(We) / Re if np.isfinite(Re) and Re != 0 else 0.0
    De = Ca / Re if np.isfinite(Re) and Re != 0 else 0.0
    Ec = np.sqrt(We) / Ca if np.isfinite(Ca) and Ca != 0 else 0.0

    return Scales(Lc=Lc, tc=tc, Uc=Uc, pc=pc, Ca=Ca, Re=Re, We=We, Oh=Oh, De=De, Ec=Ec)


def rescale_radial_history(t, R, Rd, Rdd, R_from: float, R_to: float):
    """Convert a radial history nondimensionalised by ``R_from`` to one by ``R_to``.

    The IMR radial solvers scale lengths by the maximum radius while the
    perturbation solver scales by the equilibrium radius, so histories must be
    converted before they are handed over.  With ``k = R_from / R_to``:

        R   -> k R          t   -> k t
        Rd  -> Rd           Rdd -> Rdd / k

    ``Rd`` is unchanged because both length and time rescale by the same factor.
    The legacy drivers performed this inline and, in both copies, left ``Rd``
    untouched while scaling ``R``, ``Rdd`` and ``t`` -- which is what this
    reproduces.
    """
    k = R_from / R_to
    return (np.asarray(t) * k, np.asarray(R) * k, np.asarray(Rd), np.asarray(Rdd) / k)
