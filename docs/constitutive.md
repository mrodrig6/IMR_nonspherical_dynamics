# Constitutive models: what works, and what generalising them requires

## Where the material model lives

The solver's coefficients were derived by hand for one material model and then
expanded symbolically. The result is that the model is spread across seven
places in the original MATLAB, with no derivation recorded anywhere:

| MATLAB source | What it holds |
| --- | --- |
| `f_epsilon_coeffs.m` | `elast`, `sselast` — the wall-to-infinity stress integral in the amplitude equation. This is where the `2F1` functions come from. |
| `f_non_int_coeffs.m` | `Ct3`, `Ct5`, `Ct7` — pointwise stress terms in the `T` equation |
| `f_bulk_kappa.m`, `f_bulk_KJ.m` | `Ckap`, `CJ`, `CK` — stress inside the integral kernels |
| `f_Teq_epsilon_coefs.m` | `epCoef` — stress coupling the amplitude into the `T` rows |
| `f_source_terms.m` | `Sp` — the stress source to the toroidal field |
| `f_boundary_conditions.m` | `zeta` — the elastic wall traction |

All seven carry the same signature, a `(-1 + 3*alpha)` term paired with an
`alpha` term. That pairing is exactly

```
W  = (G/2) [ (I1 - 3) + alpha (I1 - 3)^2 ]
W' = (G/2) [ 1 + 2 alpha (I1 - 3) ]
W''= G alpha
```

so the model is a quadratic (Yeoh) strain energy in parallel with a Newtonian
dashpot. Adding a material today means re-deriving and re-expanding all seven
by hand.

## Why `W(I1)` is enough

For a spherically symmetric base flow in an incompressible medium the
deformation at radius `r` is fixed by a single stretch,

```
lam = r / r0,     r0 = (r^3 - R^3 + Req^3)^(1/3),
```

giving principal stretches `(lam^-2, lam, lam)` and invariants

```
I1 = lam^-4 + 2 lam^2,      I2 = 2 lam^-2 + lam^4.
```

Linearising the stress about that base state involves the strain energy only
through `W'(I1)` and `W''(I1)`. That is why a model can be specified by two
scalar functions, and it is what the interface in
`src/imrns/constitutive/base.py` asks for.

## Stage A — shipped

The interface exists and the quadratic family sits behind it, using the
original closed forms unchanged. Available now, exactly:

- `QuadraticLaw(G, alpha)` — the model the solver was derived for
- `NeoHookean(G)` — the `alpha = 0` member
- `Newtonian(mu)` — the `G = 0` limit
- `KelvinVoigt(elastic, mu)` — any of the above with a dashpot
- the inviscid limit, `mu = 0`

Numerics are bit-identical to the MATLAB path. There is no scientific risk in
Stage A; it is a refactor of how the model is *named*, not of how it is
*evaluated*.

A model outside the family raises `ModelNotSupported` from
`quadratic_form()`. `Gent` is defined — its `W`, `dW` and `d2W` are correct and
usable for anything that needs the strain energy — but it is rejected by the
solver, because evaluating it with the quadratic-law coefficients would produce
confident, wrong numbers.

## Stage B — the general case

Two kinds of coefficient need generalising, and they are not equally hard.

**Pointwise terms** (`Ct3`, `Ct5`, `Ct7`, `Ckap`, `CJ`, `CK`, `epCoef`, `Sp`,
`zeta`) are already evaluated on the radial grid. They follow from `dW` and
`d2W` by the chain rule, with `dlam/dr` known analytically from
incompressibility. JAX supplies the derivatives exactly, which is a second,
independent reason for the JAX backend to exist.

**Integrated terms** (`elast`, `sselast`) are the hard ones. They are
`int_R^inf (...) dr` evaluated in closed form, which is why `2F1` appears.
Generalising means replacing the closed form with quadrature on the existing
mapped grid. That removes the need for a new hypergeometric derivation per
model and is what makes arbitrary `W(I1)` tractable at all.

### The gate

The generic path must reproduce the hand-expanded closed forms to about `1e-10`
for both `alpha = 0` and `alpha != 0`. That test is the whole safety argument,
and `src/imrns/constitutive/` plus `tests/test_kernels_vs_matlab.py` already
provide the reference side of it: the existing coefficients are verified
against the MATLAB source text at `rtol=1e-13`, so they are a trustworthy
oracle.

If the generic derivation disagrees with them, one of the two is wrong, and
that is worth knowing either way.

Once the gate passes, these become available with no further derivation:

- Gent, `W = -(G/2) Jm ln(1 - (I1-3)/Jm)` — already defined
- Fung, Arruda-Boyce, and any other `I1`-only strain energy
- user-supplied `W(I1)`, differentiated automatically

## Outside this interface

**`I2`-dependent models**, Mooney-Rivlin among them. The current derivation
assumes `W(I1)` only. `I2` is a different reduction and needs its own
linearisation; it is not a matter of adding a term.

**Rate-type viscoelasticity** — Oldroyd-B, Giesekus, Zener. These carry their
own stress evolution equation, so they add state variables to the ODE system
and contribute extra blocks to the operator. The interface is designed to admit
this (a model may return additional state blocks) but nothing implements it
yet. This is the right shape for the extension, not a promise that it is nearly
done.

**Generalised-Newtonian viscosity** — Carreau, power law. The implemented
viscous terms assume a constant `mu` entering only through `1/Re`. A
strain-rate-dependent viscosity changes the structure of those terms, not just
their value.
