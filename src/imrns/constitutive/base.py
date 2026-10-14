"""The constitutive-model interface.

Everything the solver needs to know about the surrounding material passes
through this module.  A model supplies two things:

* a **strain-energy function** ``W(I1)`` for the incompressible elastic
  response, through its first two derivatives ``dW`` and ``d2W``;
* a **viscosity**, currently Newtonian.

Why only ``I1``
---------------
For a spherically symmetric base flow in an incompressible medium the
deformation at radius ``r`` is fixed by the single stretch

    lam = r / r0,      r0 = (r^3 - R^3 + Req^3)^(1/3),

so the principal stretches are ``(lam^-2, lam, lam)`` and the invariants reduce
to functions of ``lam`` alone:

    I1 = lam^-4 + 2 lam^2,      I2 = 2 lam^-2 + lam^4.

The linearised perturbation stress therefore depends on the base state only
through ``W'(I1)`` and ``W''(I1)``.  That is the whole reason a model can be
specified by two scalar functions.

What is implemented today
-------------------------
The coefficient expressions the solver uses were derived by hand for the
quadratic (Yeoh) law

    W = (G/2) [ (I1 - 3) + alpha (I1 - 3)^2 ],

which has ``W' = (G/2)(1 + 2 alpha (I1-3))`` and ``W'' = G alpha``.  That is
exactly the ``(-1 + 3 alpha)`` / ``alpha`` pairing visible throughout
:mod:`imrns.assemble.kernels`.  Any model in that family -- neo-Hookean
(``alpha = 0``), the quadratic law, and the purely viscous limit (``G = 0``) --
is supported now, exactly, with no change in numerics.

A model outside the family is rejected by :meth:`MaterialModel.quadratic_form`
with a message saying so, rather than silently evaluated with the wrong
coefficients.  Supporting general ``W(I1)`` requires re-deriving the
coefficients generically; see ``docs/constitutive.md``.

Models whose stress carries its own evolution equation -- Oldroyd-B, Giesekus,
any rate-type viscoelasticity -- need extra state variables in the ODE system
and are outside this interface entirely.
"""

from __future__ import annotations

from dataclasses import dataclass


class ModelNotSupported(NotImplementedError):
    """Raised when a model cannot be expressed in the implemented closed form."""


@dataclass(frozen=True)
class QuadraticForm:
    """The ``(G, alpha)`` pair the assembled coefficients are written in terms of."""

    G: float
    alpha: float


class MaterialModel:
    """Base class for constitutive models.

    Subclasses define the elastic strain energy through :meth:`dW` and
    :meth:`d2W`, and declare whether they fall inside the quadratic family by
    overriding :meth:`quadratic_form`.
    """

    name: str = "material"

    def __init__(self, G: float = 0.0, mu: float = 0.0):
        if G < 0 or mu < 0:
            raise ValueError("G and mu must be non-negative")
        self.G = float(G)
        self.mu = float(mu)

    # -- elastic response ---------------------------------------------------

    def W(self, I1):
        """Strain energy per unit volume at first invariant ``I1``."""
        raise NotImplementedError

    def dW(self, I1):
        """``dW/dI1``."""
        raise NotImplementedError

    def d2W(self, I1):
        """``d2W/dI1^2``."""
        raise NotImplementedError

    # -- what the assembled coefficients need -------------------------------

    def quadratic_form(self) -> QuadraticForm:
        """Return the ``(G, alpha)`` the closed-form coefficients require.

        Raises :class:`ModelNotSupported` for a model outside the quadratic
        family.
        """
        raise ModelNotSupported(
            f"{type(self).__name__} is not in the quadratic (Yeoh) family, so the "
            "closed-form coefficients implemented today do not apply to it. "
            "See docs/constitutive.md."
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(G={self.G:g}, mu={self.mu:g})"


def stretch_invariant(lam):
    """First invariant ``I1`` at local stretch ``lam`` for incompressible radial flow."""
    return lam ** (-4) + 2.0 * lam**2
