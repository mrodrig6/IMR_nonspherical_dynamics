"""Hyperelastic strain-energy models.

See :mod:`imrns.constitutive.base` for why a model is specified by ``W(I1)``
and which models the implemented coefficients cover.
"""

from __future__ import annotations

import numpy as np

from .base import MaterialModel, ModelNotSupported, QuadraticForm


class QuadraticLaw(MaterialModel):
    """Yeoh-type quadratic law, the model the solver was derived for.

        W = (G/2) [ (I1 - 3) + alpha (I1 - 3)^2 ]

    ``alpha`` controls strain stiffening: ``alpha = 0`` is neo-Hookean,
    ``alpha > 0`` stiffens at large stretch.
    """

    name = "quadratic_law"

    def __init__(self, G: float, alpha: float = 0.0, mu: float = 0.0):
        super().__init__(G=G, mu=mu)
        self.alpha = float(alpha)

    def W(self, I1):
        e = np.asarray(I1) - 3.0
        return 0.5 * self.G * (e + self.alpha * e**2)

    def dW(self, I1):
        e = np.asarray(I1) - 3.0
        return 0.5 * self.G * (1.0 + 2.0 * self.alpha * e)

    def d2W(self, I1):
        return np.full_like(np.asarray(I1, dtype=float), self.G * self.alpha)

    def quadratic_form(self) -> QuadraticForm:
        return QuadraticForm(G=self.G, alpha=self.alpha)

    def __repr__(self) -> str:
        return f"QuadraticLaw(G={self.G:g}, alpha={self.alpha:g}, mu={self.mu:g})"


class NeoHookean(QuadraticLaw):
    """Neo-Hookean solid, ``W = (G/2)(I1 - 3)``.

    The ``alpha = 0`` member of :class:`QuadraticLaw`.
    """

    name = "neo_hookean"

    def __init__(self, G: float, mu: float = 0.0):
        super().__init__(G=G, alpha=0.0, mu=mu)

    def __repr__(self) -> str:
        return f"NeoHookean(G={self.G:g}, mu={self.mu:g})"


class Gent(MaterialModel):
    """Gent model with limiting chain extensibility ``Jm``.

        W = -(G/2) Jm ln(1 - (I1 - 3)/Jm)

    Defined here because the strain energy is part of the model library, but it
    is **not** in the quadratic family, so :meth:`quadratic_form` rejects it and
    the solver will refuse to run with it until the generic coefficient path
    exists.  See ``docs/constitutive.md``.
    """

    name = "gent"

    def __init__(self, G: float, Jm: float, mu: float = 0.0):
        super().__init__(G=G, mu=mu)
        if Jm <= 0:
            raise ValueError("Jm must be positive")
        self.Jm = float(Jm)

    def W(self, I1):
        return -0.5 * self.G * self.Jm * np.log(1.0 - (np.asarray(I1) - 3.0) / self.Jm)

    def dW(self, I1):
        return 0.5 * self.G / (1.0 - (np.asarray(I1) - 3.0) / self.Jm)

    def d2W(self, I1):
        return 0.5 * self.G / self.Jm / (1.0 - (np.asarray(I1) - 3.0) / self.Jm) ** 2

    def __repr__(self) -> str:
        return f"Gent(G={self.G:g}, Jm={self.Jm:g}, mu={self.mu:g})"


__all__ = ["Gent", "ModelNotSupported", "NeoHookean", "QuadraticLaw"]
