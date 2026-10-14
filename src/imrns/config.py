"""Simulation configuration.

Replaces ``matlab_legacy/examples/example_config.m`` and the eighteen
three-line scripts that called it.  A case is a plain dataclass, loadable from
TOML, so adding a case means adding a file rather than a script.

The dimensional material properties live here; :func:`imrns.scales.compute_scales`
turns them into the dimensionless groups the solver uses.  That conversion
happened inline, twice, in the MATLAB drivers.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .radial import RadialHistory, constant_radius, sinusoidal_radius
from .scales import Scales, compute_scales

__all__ = ["PRESETS", "CaseConfig", "Drive", "Material", "Numerics", "load_case"]


@dataclass
class Material:
    """Dimensional properties of the surrounding medium (SI units)."""

    rho: float = 1048.0
    p_inf: float = 101325.0
    sigma: float = 0.056
    mu: float = 0.015
    G: float = 2.0e3
    alpha: float = 0.1

    def scales(self, Req: float) -> Scales:
        return compute_scales(Req, self.rho, self.p_inf, self.mu, self.G, self.sigma)


@dataclass
class Drive:
    """How the bubble radius is driven.

    ``kind`` is one of:

    ``"free"``
        Constant radius.  The perturbation evolves with no radial forcing, and
        the solver's operator is time invariant.
    ``"sinusoidal"``
        Analytic oscillation about ``R0``, for exercising the moving-grid terms
        without an external radial solver.
    ``"imrv2"``
        A history supplied from the IMRv2 radial solver; see
        :mod:`imrns.radial.imrv2`.  Not generated here.
    """

    kind: str = "free"
    Rmax: float = 50e-6
    Req: float = 50e-6
    t_end: float = 12.0
    n_steps: int = 3000
    amplitude: float = 0.1
    omega: float = 2.0 * np.pi


@dataclass
class Numerics:
    """Discretisation and solver options."""

    xN: int = 256
    L: float = 5.0
    model: str = "me"
    closure: str = "rot"
    method: int = 2
    backend: str = "numpy"


@dataclass
class CaseConfig:
    """A complete, runnable case."""

    name: str = "case"
    description: str = ""
    modes: list[int] = field(default_factory=lambda: [8])
    ep0: list[float] = field(default_factory=lambda: [0.1])
    epd0: list[float] = field(default_factory=lambda: [0.0])
    material: Material = field(default_factory=Material)
    drive: Drive = field(default_factory=Drive)
    numerics: Numerics = field(default_factory=Numerics)

    def scales(self) -> Scales:
        return self.material.scales(self.drive.Req)

    def radial_history(self) -> RadialHistory:
        """Build the radial history this case asks for.

        Raises for ``kind = "imrv2"``, which needs a history from outside.
        """
        d = self.drive
        if d.kind == "free":
            return constant_radius(d.t_end, d.n_steps, R=1.0)
        if d.kind == "sinusoidal":
            return sinusoidal_radius(d.t_end, d.n_steps, d.amplitude, d.omega, R0=1.0)
        if d.kind == "imrv2":
            raise ValueError(
                f"case {self.name!r} needs a radial history from IMRv2. Compute it there, "
                "convert it with imrns.radial.imrv2.adapt_history, and pass it to "
                "run_case(history=...)."
            )
        raise ValueError(f"unknown drive kind {d.kind!r}; use free, sinusoidal or imrv2")

    def solver_kwargs(self) -> dict:
        """Everything :func:`imrns.solver.solve_perturbation` needs but the history."""
        s = self.scales()
        return dict(
            modes=self.modes,
            Ca=s.Ca, Re=s.Re, We=s.We,
            alpha=self.material.alpha,
            xN=self.numerics.xN, L=self.numerics.L,
            ep0=self.ep0, epd0=self.epd0,
            model=self.numerics.model, closure=self.numerics.closure,
            method=self.numerics.method, backend=self.numerics.backend,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _merge(cls, data: dict):
    known = {f for f in cls.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def load_case(path) -> CaseConfig:
    """Read a case from a TOML file.

    Unknown keys are rejected rather than ignored, so a typo in a parameter
    name fails immediately instead of silently running the default.
    """
    path = Path(path)
    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    material = _merge(Material, raw.pop("material", {}))
    drive = _merge(Drive, raw.pop("drive", {}))
    numerics = _merge(Numerics, raw.pop("numerics", {}))

    modes = list(raw.pop("modes", [8]))
    ep0 = list(np.atleast_1d(raw.pop("ep0", [0.1])))
    epd0 = list(np.atleast_1d(raw.pop("epd0", [0.0] * len(modes))))
    if len(ep0) == 1 and len(modes) > 1:
        ep0 = ep0 * len(modes)
    if len(epd0) == 1 and len(modes) > 1:
        epd0 = epd0 * len(modes)
    if not (len(ep0) == len(epd0) == len(modes)):
        raise ValueError(f"{path.name}: ep0 and epd0 must have one value per mode")

    name = raw.pop("name", path.stem)
    description = raw.pop("description", "")
    if raw:
        raise ValueError(f"{path.name}: unknown top-level keys: {sorted(raw)}")

    return CaseConfig(name=name, description=description, modes=modes, ep0=ep0, epd0=epd0,
                      material=material, drive=drive, numerics=numerics)


# Material presets matching the legacy example_config material types.
PRESETS = {
    "viscous": Material(mu=0.015, G=0.0, alpha=0.0),
    "elastic": Material(mu=0.0, G=2.0e3, alpha=0.1),
    "viscoelastic": Material(mu=0.015, G=2.0e3, alpha=0.1),
}
