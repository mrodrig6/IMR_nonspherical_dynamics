"""Assembly of the per-mode operator block.

:mod:`imrns.assemble.kernels` holds the physics -- the coefficient expressions
ported from the MATLAB source -- as pure arithmetic, so a single copy serves
all three backends.  The backend modules differ only in how those coefficients
are written into the matrix:

* :mod:`imrns.assemble.numpy_backend` -- the reference implementation
* :mod:`imrns.assemble.numba_backend` -- the same assembly with compiled kernels
* :mod:`imrns.assemble.jax_backend` -- functional writes, traceable and
  differentiable

Backends are selected through :func:`imrns.backends.get_backend` rather than
imported directly, so that an optional dependency is only imported when asked
for.
"""
