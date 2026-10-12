# IMR Nonspherical Dynamics

Small-amplitude nonspherical perturbations of an inertial microcavitation
bubble in a soft viscoelastic material.

Given a radial history `R(t)` of a spherical bubble, the solver evolves the
spherical-harmonic amplitudes of the interface,

```
r_wall(theta, t) = R(t) + sum_n eps_n(t) Y_n(theta),
```

together with the rotational (toroidal) field `T_n(r, t)` that each mode drives
in the surrounding medium. Two closures are available: `rot` solves the coupled
amplitude-and-field system, `irr` keeps only the potential-flow amplitude
equation.

This is a Python port of the original MATLAB implementation, which is preserved
unmodified in [`matlab_legacy/`](matlab_legacy/) for provenance and is no longer
maintained. [`docs/porting-notes.md`](docs/porting-notes.md) records what
changed and why.

## Install

```bash
pip install -e '.[all]'      # numpy, scipy, numba, jax, matplotlib, pytest
pip install -e .             # numpy and scipy only; the numpy backend
```

Python 3.10 or newer. There is nothing to compile: the Gauss hypergeometric
function now comes from SciPy, so the MEX binaries and their Cephes C sources
are gone.

## Quick start

```python
import numpy as np
from imrns import solve_perturbation, compute_scales
from imrns.radial import constant_radius

scales = compute_scales(Req=50e-6, rho=1048, p_inf=101325,
                        mu=0.015, G=2e3, sigma=0.056)

history = constant_radius(t_end=12.0, n_steps=3000)

result = solve_perturbation(
    modes=[5, 8, 11],
    ep0=[0.075, -0.085, 0.05],
    Ca=scales.Ca, Re=scales.Re, We=scales.We, alpha=0.1,
    xN=256,
    **history.as_kwargs(),
)

print(result.ep.shape)      # (3, 3000) -- one amplitude history per mode
print(result.T.shape)       # (3, 256, 3000) -- the rotational field
```

Or from the command line:

```bash
imrns list                                            # the bundled cases
imrns info examples/free_viscoelastic_multi.toml      # parameters and groups
imrns run  examples/free_viscoelastic_multi.toml -o out.npz
```

## The model

The medium is incompressible, so the deformation at radius `r` is fixed by one
stretch,

```
lam = r / r0,      r0 = (r^3 - R^3 + Req^3)^(1/3),
```

and the strain invariant reduces to `I1 = lam^-4 + 2 lam^2`. The implemented
coefficients were derived for a Kelvin-Voigt solid with a quadratic (Yeoh)
strain energy,

```
W = (G/2) [ (I1 - 3) + alpha (I1 - 3)^2 ],
```

in parallel with a Newtonian dashpot. `alpha = 0` is neo-Hookean; `alpha > 0`
stiffens at large stretch.

Numerically: the domain `r in [R, inf)` is mapped to `x in [-1, 1)` by
`r = R(1 + L(1+x)/(1-x))`; space is discretised with second-order finite
differences and trapezoidal quadrature for the integral kernels; time with
backward Euler on the first step and variable-step BDF2 thereafter.

### Nondimensionalisation

Lengths by the equilibrium radius, pressure by ambient. Everything is derived
in one place, [`imrns/scales.py`](src/imrns/scales.py):

| Group | Definition | Meaning |
| --- | --- | --- |
| `Ca` | `pc / G` | inertia over elasticity |
| `Re` | `Lc sqrt(rho pc) / mu` | inertia over viscosity |
| `We` | `pc Lc / (2 sigma)` | inertia over surface tension |

Infinite values are fine and mean the corresponding stress vanishes: `G = 0`
gives `Ca = inf`. The solver uses only reciprocals internally, so the limits are
exact rather than approximated by a large finite number.

One exception is rejected rather than silently producing nonsense: with `Ca` and
`Re` *both* infinite there is no deviatoric stress at all, the tangential-traction
wall condition becomes identically zero, and the rotational system is singular.
Use `closure="irr"` there.

## Backends

One statement of the physics, three ways to execute it. All three are required
to agree to `1e-12` by [`tests/test_backends.py`](tests/test_backends.py).

| Backend | Use it for |
| --- | --- |
| `numpy` | the default and the reference; always importable |
| `numba` | production runs; compiles the coefficient kernels |
| `jax` | gradients, parameter sensitivity, GPU |

```python
result = solve_perturbation(..., backend="numba")
```

Numba and JAX cannot be combined — a `@njit` function cannot be called inside a
JAX trace — which is why they are alternative backends rather than layered
accelerations.

### Performance

Measured on this repository's benchmark, four modes at `xN = 256`, 200 steps.
The headline change is structural rather than compiler-driven: **the operator is
block diagonal by mode** and nobody was exploiting it. Every write the assembly
performs lands inside its own mode's block, so four 514-cubed factorisations
replace one 2056-cubed one.

| | ms/step |
| --- | --- |
| monolithic sparse solve, as the MATLAB code assembled it | 85 |
| per-mode blocks, NumPy | 31.5 |
| per-mode blocks, Numba | 26.5 |
| constant radius, LU factorisation reused | 1.3 |

Be clear about where the time goes. At `xN = 256` the per-block cost is about
6.5 ms to assemble and 2.5 ms to factorise and solve — **assembly dominates**.
Numba makes the isolated coefficient kernels about fifty times faster, but they
are a small part of assembly, so end to end it contributes about 20%. Larger
gains came from using slices rather than fancy indexing for the sub-block
writes (2.1x) and from hoisting the mode-independent advection operator out of
the mode loop.

Two optimisations were measured and rejected: banded-Schur elimination of the
`T` block was 9x *slower* (the dense coupling makes the complement a wash), and
reusing the previous step's LU as a GMRES preconditioner gained only ~1.5x at
small drift.

## Automatic differentiation

The timestep is `q^{n+1} = solve(beta0 I - dt A(theta), rhs(q^n, theta))`, which
is differentiable end to end. One reverse-mode pass gives every parameter
sensitivity at once, exactly — no finite-difference step to choose.

```python
import jax, jax.numpy as jnp
from imrns.sensitivity import amplitude_trajectory, default_parameters

def loss(params):
    ep = amplitude_trajectory(params, t=t, R=R, Rd=Rd, Rdd=Rdd, mode=5, xN=64)
    return jnp.sum((ep - measured) ** 2)

grads = jax.grad(loss)(default_parameters(Ca=50.0, Re=35.0, We=45.0, alpha=0.1))
```

Two details make this work, both verified against finite differences in
[`tests/test_gradients.py`](tests/test_gradients.py):

- **`2F1` is not in JAX.** The elastic coefficients evaluate it at
  `z = 1 - sr^-3`, and `sr = R/Req` is differentiable state, so the derivative
  is needed and not just the value. It is supplied through a `custom_jvp` using
  `d/dz 2F1(a,b;c;z) = (ab/c) 2F1(a+1,b+1;c+1;z)`, with values from a
  `pure_callback` to SciPy.
- **Reverse mode would store every operator.** That is tens of gigabytes for a
  production run, so the scan body is wrapped in `jax.checkpoint` and each
  operator is rebuilt during the backward pass. The gradients are identical
  with and without, which is also a test.

The differentiable path uses backward Euler throughout; use
`solve_perturbation` for production trajectories and this module for gradients.

## Constitutive models

Material behaviour enters through one interface,
[`imrns/constitutive/`](src/imrns/constitutive/). A model supplies the elastic
strain energy through `dW(I1)` and `d2W(I1)`, plus a viscosity.

```python
from imrns.constitutive import NeoHookean, QuadraticLaw, KelvinVoigt

KelvinVoigt(QuadraticLaw(G=2e3, alpha=0.1), mu=0.015)   # the default model
NeoHookean(G=2e3)                                        # alpha = 0
```

**What works today.** The hand-derived coefficients the solver uses assume a
quadratic `W(I1)`. Everything in that family is exact and available now: the
quadratic law, neo-Hookean, the Newtonian limit (`G = 0`) and the inviscid limit
(`mu = 0`). A model outside the family is rejected with an explanation rather
than silently evaluated with the wrong coefficients.

**What does not, yet.** Supporting general `W(I1)` — Gent, Fung, Arruda-Boyce,
or anything user-supplied — means re-deriving the coefficients generically and
checking them against the closed forms. `I2`-dependent models such as
Mooney-Rivlin need their own derivation, and rate-type viscoelasticity
(Oldroyd-B, Giesekus) adds state variables to the system and needs an extended
interface. [`docs/constitutive.md`](docs/constitutive.md) sets out what is
required and why the staging is what it is.

## Repository layout

```text
src/imrns/
  config.py          case definitions, TOML loading
  scales.py          nondimensionalisation, in one place
  grid.py            the x <-> r map and its metric coefficients
  operators.py       finite differences, upwinding, quadrature
  layout.py          state-vector layout; replaces f_get_indicies
  special.py         2F1, from SciPy
  solver.py          the time loop
  stepper.py         BDF coefficients, block solve, factorisation cache
  constitutive/      material models and the interface they implement
  assemble/
    kernels.py       the physics: one copy, shared by all three backends
    numpy_backend.py numba_backend.py  jax_backend.py
  radial/            radial histories; IMRv2 bridge
  sensitivity.py     differentiable trajectories
  colormaps.py       the colormaps the MATLAB tree referenced but never shipped
  cli.py
examples/            18 cases as TOML, replacing 18 three-line scripts
tests/
matlab_legacy/       the original MATLAB, unmodified and unmaintained
```

## Radial histories

The perturbation solver never computes the radial motion; it takes
`(t, R, Rd, Rdd)` as given. Built in: `constant_radius` (what the `free` cases
use) and `sinusoidal_radius`. For a history from the IMRv2 radial solver:

```python
from imrns.radial.imrv2 import adapt_history
history = adapt_history(t, R, Rd, Rdd, Rmax=150e-6, Req=50e-6)
```

`adapt_history` takes plain arrays, so it is not coupled to any particular
IMRv2 version. It is also the only place the scaling conversion happens — IMRv2
normalises lengths by the maximum radius, this solver by the equilibrium radius,
and the MATLAB drivers did that conversion inline, twice. The `lic_*` cases ask
for such a history and say so if run without one.

## Tests

```bash
pip install -e '.[all]'
PYTHONPATH=src pytest
```

The suite is built around the fact that there is no written derivation of the
model — the hand-expanded MATLAB expressions are the only statement of it. So
the port is checked from several independent directions:

- **Against the MATLAB source text.**
  [`tests/matlab_oracle.py`](tests/matlab_oracle.py) reads `matlab_legacy/*.m`,
  extracts the right-hand side of a named assignment, rewrites MATLAB array
  syntax as Python and evaluates it. Every coefficient kernel is compared
  against its own source at `rtol=1e-13`, so transcription is out of the trust
  chain entirely.
- **Against known physics.** A free inviscid bubble must reproduce the Lamb
  capillary frequency `omega^2 = (n-1)(n+1)(n+2) sigma / (rho R^3)`; `rot` must
  converge to `irr` at first order as the deviatoric stress vanishes; the
  solution must be exactly linear in the initial amplitude; modes solved
  together must match modes solved alone (the property the block-diagonal solve
  relies on).
- **Across backends and against finite differences**, as described above.
- **Against pinned trajectories**, so refactors cannot drift silently.

Neither MATLAB nor Octave is used or required. What this cannot do is prove
agreement with *your* saved MATLAB runs; if you have `.mat` outputs from a
known-good run, they can be added as fixtures.

## Defects found in the MATLAB code

Found while porting, listed so published results can be checked.
[`docs/porting-notes.md`](docs/porting-notes.md) has the details.

Two change results:

1. `s_basic_simulation.m:13` — `Rmax = 150-6`, almost certainly `150e-6`. As
   written the driver's bubble is 144 metres, and since `Lc = Req = Rmax/1.2`
   every dimensionless group in it is wrong.
2. `s_basic_simulation.m:100` — `solverOptions` is unconditionally overwritten
   with `{'Verbose', true}` immediately after the `forcedep == 'T'` branch fills
   it, so prescribed-amplitude runs from that driver could not work.

Others: the derivative stencils and the advection operator used different grid
spacings on the same grid; the quasi-equilibrium branch truncated the amplitude
histories but returned `t`, `R` and `Rd` at full length; `cmap/` was referenced
by `setup_paths`, the README and every example but never committed, so any
example with snapshot output failed.

One behaviour was checked with the author and **deliberately preserved**: in
`f_ep_source_terms`, the `V` rows are captured before `f_modify_A_bulk_map` adds
the moving-grid advection, so the second-time-derivative substitution uses the
uncorrected operator. The ordering is load-bearing and is commented as such in
both the NumPy and JAX backends.

## Not in this port yet

The 2769-line visualisation module
(`matlab_legacy/common/make_axisym_displacement_movie_all_fields.m`) — strain,
strain-rate and stress field rendering — has not been ported. It is a separate
piece of work, and the physics and rendering parts of it want separating rather
than transcribing.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

GNU General Public License v3.0 only. See [`LICENSE`](LICENSE).
