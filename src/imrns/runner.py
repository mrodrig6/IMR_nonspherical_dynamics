"""Run a configured case end to end."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CaseConfig
from .radial import RadialHistory
from .scales import Scales
from .solver import PerturbationResult, solve_perturbation

__all__ = ["CaseResult", "run_case"]


@dataclass
class CaseResult:
    """Everything one case produced."""

    config: CaseConfig
    scales: Scales
    rotational: PerturbationResult
    irrotational: PerturbationResult | None = None

    def save(self, path):
        """Write the trajectories to a compressed ``.npz``."""
        rot = self.rotational
        data = dict(t=rot.t, R=rot.R, Rd=rot.Rd, ep=rot.ep, epd=rot.epd,
                    T=rot.T, Td=rot.Td, modes=np.asarray(rot.modes))
        if self.irrotational is not None:
            data["ep_irrotational"] = self.irrotational.ep
        np.savez_compressed(path, **data)
        return path


def run_case(config: CaseConfig, history: RadialHistory | None = None,
             compare_irrotational: bool = True, progress: bool = False) -> CaseResult:
    """Solve one case, optionally alongside the irrotational comparison.

    Parameters
    ----------
    config
        The case to run.
    history
        Radial history.  Built from ``config.drive`` when omitted, which is not
        possible for ``kind = "imrv2"`` -- pass one in for those.
    compare_irrotational
        Also run the irrotational closure, which is what the MATLAB driver
        plotted alongside the rotational result.  Cheap: the irrotational
        system is two unknowns per mode.
    progress
        Print a percentage counter.
    """
    if history is None:
        history = config.radial_history()

    kwargs = config.solver_kwargs()
    rotational = solve_perturbation(**kwargs, **history.as_kwargs(), progress=progress)

    irrotational = None
    if compare_irrotational and config.numerics.closure != "irr":
        irrotational = solve_perturbation(
            **{**kwargs, "closure": "irr"}, **history.as_kwargs()
        )

    return CaseResult(config=config, scales=config.scales(),
                      rotational=rotational, irrotational=irrotational)
