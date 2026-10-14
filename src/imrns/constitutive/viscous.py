"""Viscous models.

The implemented coefficients assume a Newtonian deviatoric stress, which enters
only through ``1/Re``.  Shear-thinning models would make the effective viscosity
a function of the local strain rate and change the structure of the viscous
terms, so they are not available yet; see ``docs/constitutive.md``.
"""

from __future__ import annotations

from .base import MaterialModel, QuadraticForm


class Newtonian(MaterialModel):
    """Newtonian fluid, ``tau = 2 mu D``, with no elastic response."""

    name = "newtonian"

    def __init__(self, mu: float):
        super().__init__(G=0.0, mu=mu)

    def W(self, I1):
        return 0.0 * I1

    def dW(self, I1):
        return 0.0 * I1

    def d2W(self, I1):
        return 0.0 * I1

    def quadratic_form(self) -> QuadraticForm:
        return QuadraticForm(G=0.0, alpha=0.0)

    def __repr__(self) -> str:
        return f"Newtonian(mu={self.mu:g})"


class KelvinVoigt(MaterialModel):
    """Elastic solid in parallel with a Newtonian dashpot.

    This is the model the whole solver is built around: the stresses add, so
    the elastic part contributes through ``1/Ca`` and the viscous part through
    ``1/Re`` independently.  Supply any elastic model as ``elastic``.
    """

    name = "kelvin_voigt"

    def __init__(self, elastic: MaterialModel, mu: float):
        super().__init__(G=elastic.G, mu=mu)
        self.elastic = elastic

    def W(self, I1):
        return self.elastic.W(I1)

    def dW(self, I1):
        return self.elastic.dW(I1)

    def d2W(self, I1):
        return self.elastic.d2W(I1)

    def quadratic_form(self) -> QuadraticForm:
        return self.elastic.quadratic_form()

    def __repr__(self) -> str:
        return f"KelvinVoigt({self.elastic!r}, mu={self.mu:g})"


__all__ = ["KelvinVoigt", "Newtonian"]
