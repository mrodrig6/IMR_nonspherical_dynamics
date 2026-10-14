"""Numba-accelerated assembly.

The coefficient kernels in :mod:`imrns.assemble.kernels` are single arithmetic
expressions over length-``xN`` arrays, each building dozens of intermediate
temporaries.  NumPy allocates and traverses every one of them; Numba fuses the
whole expression into a single loop.  Measured on ``Ct7`` at ``xN = 256``:
58.8 us under NumPy, 1.2 us compiled -- about fifty times faster.

That matters only once the linear solve has been reduced to per-mode blocks.
In the original monolithic formulation the solve was roughly 97% of the
runtime and no amount of kernel acceleration would have shown up.

Only the kernels are compiled.  The surrounding assembly is matrix algebra that
already runs in BLAS, so it reuses :mod:`imrns.assemble.numpy_backend`
unchanged, with the compiled kernels injected.  There is one implementation of
the assembly logic, not two.

First call to each kernel pays a compilation cost of a second or so; set
``NUMBA_DISABLE_JIT=1`` to bypass compilation when debugging.
"""

from __future__ import annotations

import types

from numba import njit

from ..special import hyp2f1_values
from . import kernels as _k
from . import numpy_backend as _np_backend

__all__ = [
    "add_moving_grid_terms",
    "apply_boundary_conditions",
    "apply_boundary_rhs",
    "assemble_block",
    "ep_feedback_rows",
    "hyp2f1_values",
]

_JIT = dict(cache=True, fastmath=True, nogil=True)

# Compiled counterparts of every kernel the assembly calls.  fastmath is safe
# here because the reciprocal groups (inv_Ca, inv_Re, inv_We) are finite by
# construction -- see the module docstring of imrns.assemble.kernels.
_compiled = types.SimpleNamespace(
    non_int_coeffs=njit(**_JIT)(_k.non_int_coeffs),
    kappa_coeffs=njit(**_JIT)(_k.kappa_coeffs),
    KJ_coeffs=njit(**_JIT)(_k.KJ_coeffs),
    Teq_epsilon_coeffs=njit(**_JIT)(_k.Teq_epsilon_coeffs),
    epsilon_coeffs_me=njit(**_JIT)(_k.epsilon_coeffs_me),
    epsilon_coeffs_pros=njit(**_JIT)(_k.epsilon_coeffs_pros),
    forced_source=njit(**_JIT)(_k.forced_source),
    ep_feedback_pointwise=njit(**_JIT)(_k.ep_feedback_pointwise),
    ep_feedback_kappa=njit(**_JIT)(_k.ep_feedback_kappa),
    ep_feedback_J=njit(**_JIT)(_k.ep_feedback_J),
    ep_feedback_K=njit(**_JIT)(_k.ep_feedback_K),
    wall_traction=njit(**_JIT)(_k.wall_traction),
    wall_kappa_terms=njit(**_JIT)(_k.wall_kappa_terms),
)


def assemble_block(*args, **kwargs):
    """Assemble one mode's operator block using the compiled kernels."""
    kwargs.setdefault("kern", _compiled)
    return _np_backend.assemble_block(*args, **kwargs)


def ep_feedback_rows(*args, **kwargs):
    """Add the rotational feedback rows using the compiled kernels."""
    kwargs.setdefault("kern", _compiled)
    return _np_backend.ep_feedback_rows(*args, **kwargs)


def apply_boundary_conditions(*args, **kwargs):
    """Apply boundary conditions using the compiled kernels."""
    kwargs.setdefault("kern", _compiled)
    return _np_backend.apply_boundary_conditions(*args, **kwargs)


def apply_boundary_rhs(*args, **kwargs):
    """Apply the right-hand-side boundary terms using the compiled kernels."""
    kwargs.setdefault("kern", _compiled)
    return _np_backend.apply_boundary_rhs(*args, **kwargs)


add_moving_grid_terms = _np_backend.add_moving_grid_terms
upwind_advection = _np_backend.upwind_advection
