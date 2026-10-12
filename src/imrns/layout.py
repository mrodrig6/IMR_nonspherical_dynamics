"""State-vector layout for the coupled modal system.

Each spherical-harmonic mode owns a contiguous block of the state vector.  What
goes in that block depends on three switches:

* ``closure`` -- ``"rot"`` carries the rotational field, ``"irr"`` drops it.
* ``model`` -- ``"me"`` evolves ``T`` and ``V = dT/dt|_r`` as separate unknowns;
  ``"pros"`` evolves ``T`` alone.
* ``forced_ep`` -- when the interface amplitude is prescribed, ``eps`` and
  ``eps'`` leave the state vector.

The resulting block sizes:

==========  =========  ==========  ====================  ==========
closure     model      forced_ep   block contents        block size
==========  =========  ==========  ====================  ==========
``irr``     any        any         eps, eps'             2
``rot``     ``me``     yes         T, V                  2 xN
``rot``     ``me``     no          T, V, eps, eps'       2 xN + 2
``rot``     ``pros``   yes         T                     xN
``rot``     ``pros``   no          T, eps, eps'          xN + 2
==========  =========  ==========  ====================  ==========

The legacy ``f_get_indicies`` recomputed these offsets inside every assembly
routine, once per mode per timestep.  They are fixed for a whole run, so they
are computed once here.

**The global operator is block diagonal.**  Every write performed by the
assembly routines lands inside the writing mode's own block -- the linearised
modes do not couple.  :attr:`StateLayout.block_size` is therefore the size of
the systems actually solved; see :mod:`imrns.stepper`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

VALID_MODELS = ("me", "pros")
VALID_CLOSURES = ("rot", "irr")


@dataclass(frozen=True)
class ModeIndices:
    """Positions within the *global* state vector for one mode."""

    T: np.ndarray | None
    V: np.ndarray | None
    ep: int | None
    epd: int | None
    base: int
    block_size: int

    @property
    def block(self) -> slice:
        """Slice covering this mode's whole block."""
        return slice(self.base, self.base + self.block_size)

    @property
    def T_sl(self) -> slice:
        """``T`` indices as a slice.

        The index arrays are contiguous ranges, so a slice addresses the same
        entries while giving a view rather than a fancy-indexing copy.  For a
        256-point grid that makes each sub-block update about six times
        cheaper, and assembly dominates the runtime at that size.
        """
        if self.T is None:
            raise AttributeError("this layout has no T block")
        return slice(int(self.T[0]), int(self.T[-1]) + 1)

    @property
    def V_sl(self) -> slice:
        """``V`` indices as a slice; see :attr:`T_sl`."""
        if self.V is None:
            raise AttributeError("this layout has no V block")
        return slice(int(self.V[0]), int(self.V[-1]) + 1)


@dataclass(frozen=True)
class StateLayout:
    """Index bookkeeping for the whole modal system."""

    modes: tuple[int, ...]
    xN: int
    model: str
    closure: str
    forced_ep: bool
    block_size: int
    system_size: int
    per_mode: tuple[ModeIndices, ...] = field(repr=False)

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    @property
    def has_field(self) -> bool:
        """True when the rotational field ``T`` is part of the state."""
        return self.closure == "rot"

    @property
    def has_V(self) -> bool:
        """True when ``V = dT/dt|_r`` is a separate unknown."""
        return self.closure == "rot" and self.model == "me"

    @property
    def has_amplitude(self) -> bool:
        """True when ``eps`` and ``eps'`` are solved for rather than prescribed."""
        return not self.forced_ep

    def local(self, i: int) -> ModeIndices:
        """Indices for mode ``i`` rebased to the start of its own block."""
        m = self.per_mode[i]
        off = m.base
        return ModeIndices(
            T=None if m.T is None else m.T - off,
            V=None if m.V is None else m.V - off,
            ep=None if m.ep is None else m.ep - off,
            epd=None if m.epd is None else m.epd - off,
            base=0,
            block_size=m.block_size,
        )


def build_layout(modes, xN: int, model: str, closure: str, forced_ep: bool) -> StateLayout:
    """Construct the :class:`StateLayout` for a run.

    Replaces ``matlab_legacy/common/f_get_indicies.m``.
    """
    model = str(model).lower()
    closure = str(closure).lower()
    if model not in VALID_MODELS:
        raise ValueError(f"model must be one of {VALID_MODELS}, got {model!r}")
    if closure not in VALID_CLOSURES:
        raise ValueError(f"closure must be one of {VALID_CLOSURES}, got {closure!r}")

    modes = tuple(int(n) for n in np.atleast_1d(modes).ravel())
    if not modes:
        raise ValueError("at least one mode number is required")
    if any(n < 1 for n in modes):
        raise ValueError(f"mode numbers must be >= 1, got {modes}")

    if closure == "irr":
        block_size = 2
    elif model == "me":
        block_size = 2 * xN + (0 if forced_ep else 2)
    else:
        block_size = xN + (0 if forced_ep else 2)

    per_mode = []
    for i in range(len(modes)):
        base = i * block_size
        if closure == "irr":
            idx = ModeIndices(None, None, base + 0, base + 1, base, block_size)
        elif model == "me":
            T = base + np.arange(xN)
            V = base + xN + np.arange(xN)
            ep = None if forced_ep else base + 2 * xN
            epd = None if forced_ep else base + 2 * xN + 1
            idx = ModeIndices(T, V, ep, epd, base, block_size)
        else:
            T = base + np.arange(xN)
            ep = None if forced_ep else base + xN
            epd = None if forced_ep else base + xN + 1
            idx = ModeIndices(T, None, ep, epd, base, block_size)
        per_mode.append(idx)

    return StateLayout(
        modes=modes,
        xN=xN,
        model=model,
        closure=closure,
        forced_ep=forced_ep,
        block_size=block_size,
        system_size=block_size * len(modes),
        per_mode=tuple(per_mode),
    )
