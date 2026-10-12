# Third-Party Notices

## The Python package

`imrns` contains no third-party source. It depends on NumPy, SciPy and
optionally Numba, JAX and Matplotlib, all installed from their own
distributions under their own licenses.

The Gauss hypergeometric function `2F1`, which the elastic coefficients
require, now comes from `scipy.special.hyp2f1`. That removes the compiled MEX
dependency the MATLAB implementation carried, along with the vendored Cephes
sources described below.

## The legacy MATLAB tree

`matlab_legacy/` preserves the original MATLAB implementation unmodified, and
it still contains the third-party code it always did. These notices apply to
that directory only; nothing in `src/imrns/` uses any of it.

### Gauss Hypergeometric Function MATLAB/MEX Support

- Source: Siyi Deng, "Gauss hypergeometric function," MATLAB Central File
  Exchange, Version 1.0.0.0, published 11 Oct 2013.
- URL: https://www.mathworks.com/matlabcentral/fileexchange/43865-gauss-hypergeometric-function
- Files: `matlab_legacy/common/hyp2f1.m`,
  `matlab_legacy/common/make_hyp2f1/mexhyp2f1.c`,
  `matlab_legacy/common/make_hyp2f1/make_hyp2f1.m`, and the prebuilt binaries
  `mexhyp2f1.mexa64` and `mexhyp2f1.mexw64`.

### Cephes Math Library Components

Several C sources in `matlab_legacy/common/make_hyp2f1/` are based on Cephes
Math Library code by Stephen L. Moshier. The original headers carry the
applicable copyright notices and must be kept intact when modifying or
redistributing them:

- `gamma.c`
- `hyp2f1.c`
- `mconf.h`
- `polevl.c`
- `psi.c`
- `round.c`
