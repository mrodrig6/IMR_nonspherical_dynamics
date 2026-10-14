"""Nonspherical perturbation dynamics of inertial microcavitation bubbles.

Small-amplitude nonspherical perturbations of a spherical bubble collapsing or
oscillating in a soft viscoelastic medium.  Given a radial history ``R(t)``, the
solver evolves the spherical-harmonic amplitudes ``eps_n(t)`` of the interface
and the rotational field ``T_n(r, t)`` each mode drives in the surrounding
material.

    import numpy as np
    from imrns import solve_perturbation, compute_scales
    from imrns.radial import constant_radius

    scales = compute_scales(Req=50e-6, rho=1048, p_inf=101325,
                            mu=0.015, G=2e3, sigma=0.056)
    history = constant_radius(t_end=12.0, n_steps=3000)
    result = solve_perturbation(
        modes=[5, 8, 11], ep0=[0.075, -0.085, 0.05],
        Ca=scales.Ca, Re=scales.Re, We=scales.We, alpha=0.1,
        **history.as_kwargs(),
    )

See ``README.md`` for the model, the backends and the constitutive interface.
"""

from .backends import available_backends
from .colormaps import brown_centered_diverging
from .config import CaseConfig, Drive, Material, Numerics, load_case
from .runner import CaseResult, run_case
from .scales import Scales, compute_scales
from .solver import PerturbationResult, solve_perturbation

__version__ = "0.1.0"

__all__ = [
    "CaseConfig",
    "CaseResult",
    "Drive",
    "Material",
    "Numerics",
    "PerturbationResult",
    "Scales",
    "__version__",
    "available_backends",
    "brown_centered_diverging",
    "compute_scales",
    "load_case",
    "run_case",
    "solve_perturbation",
]
