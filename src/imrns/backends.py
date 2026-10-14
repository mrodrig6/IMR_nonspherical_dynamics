"""Backend selection.

Three backends share one statement of the physics
(:mod:`imrns.assemble.kernels`) and differ only in how the assembled block is
built and solved:

``numpy``
    Reference implementation.  Always available, and the one to read.
``numba``
    The same assembly with the coefficient kernels JIT-compiled.  Measured
    about 50x faster than NumPy on the coefficient evaluation, which dominates
    once the linear solve has been reduced to per-mode blocks.
``jax``
    Traceable and differentiable, for parameter sensitivities and GPU
    execution.  See :mod:`imrns.assemble.jax_backend`.

Numba and JAX cannot be mixed in one call path -- a ``@njit`` function cannot
be called inside a JAX trace -- which is why these are separate backends rather
than layered accelerations of one.
"""

from __future__ import annotations

import functools

__all__ = ["available_backends", "get_backend"]


def available_backends() -> list[str]:
    """Names of the backends importable in this environment."""
    names = ["numpy"]
    for name, module in (("numba", "numba"), ("jax", "jax")):
        try:
            __import__(module)
        except ImportError:
            continue
        names.append(name)
    return names


@functools.lru_cache(maxsize=4)
def get_backend(name: str):
    """Import and return the backend module called ``name``."""
    name = str(name).lower()
    if name == "numpy":
        from .assemble import numpy_backend

        return numpy_backend
    if name == "numba":
        try:
            from .assemble import numba_backend
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "the numba backend needs numba installed: pip install 'imrns[numba]'"
            ) from exc
        return numba_backend
    if name == "jax":
        try:
            from .assemble import jax_backend
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "the jax backend needs jax installed: pip install 'imrns[jax]'"
            ) from exc
        return jax_backend
    raise ValueError(f"unknown backend {name!r}; choose from {available_backends()}")
