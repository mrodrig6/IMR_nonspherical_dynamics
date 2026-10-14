"""Radial bubble histories ``R(t)`` that drive the perturbation solver.

The perturbation solver never computes the radial motion itself -- it takes
``(t, R, Rd, Rdd)`` as given.  Two sources are provided:

* :mod:`imrns.radial.prescribed` for analytic or user-supplied histories,
  including the constant-radius case the free examples use.
* :mod:`imrns.radial.imrv2` for a history produced by the IMRv2 radial solver.

Whatever the source, the history must be expressed in the perturbation
solver's own scaling -- lengths by the equilibrium radius -- before it is
handed over.  :func:`imrns.scales.rescale_radial_history` does that conversion.
"""

from .prescribed import RadialHistory, constant_radius, from_arrays, sinusoidal_radius

__all__ = ["RadialHistory", "constant_radius", "from_arrays", "sinusoidal_radius"]
