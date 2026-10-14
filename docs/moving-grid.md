# The moving-grid (advection) term

Short answer: it is required, and it is pure kinematics — the standard
Arbitrary Lagrangian–Eulerian (ALE) transformation for a mesh that moves. It is
not a fudge, and nothing about it is fitted or tuned. What *is* wrong is a stale
comment in the MATLAB source that describes a different (also correct)
formulation than the one implemented, which is almost certainly why the term
reads as unmotivated.

## 1. The grid moves, and that is the whole story

The solver works on

```
r(x, t) = R(t) g(x),        g(x) = 1 + L (1 + x) / (1 - x),      x in [-1, 1)
```

A node of constant `x` sits at `r = R(t) g(x)`. Because `R` depends on time, **a
node of constant `x` is not at a fixed physical location**. It is also not a
material point. So `d/dt` at constant `x` — which is what a time-stepper
advancing nodal values actually computes — is neither the Eulerian derivative
`∂/∂t|_r` nor a material derivative. The governing equations are written with
`∂/∂t|_r`. Something has to reconcile the two, and that something is this term.

## 2. Derivation

Take any field `φ`, written either on the computational grid as `φ(x, t)` or in
physical space as `φ̃(r, t)`, with `φ(x, t) = φ̃(r(x, t), t)`. The chain rule
gives

```
∂φ/∂t|_x  =  ∂φ̃/∂t|_r  +  (∂r/∂t|_x) (∂φ̃/∂r)
```

The two factors in the last term are

```
∂r/∂t|_x = Ṙ g(x) = (Ṙ/R) r
∂φ̃/∂r   = (∂φ/∂x) (∂x/∂r) = (∂φ/∂x) / (R g'(x)),     g'(x) = 2L / (1 - x)^2
```

Multiplying them,

```
(Ṙ/R) R g(x) · (1 - x)^2 / (2 R L) = (Ṙ/R) · (1 - x)^2 / (2L) · [1 + L(1+x)/(1-x)]
```

which is exactly `f_grid_velocity` / `imrns.grid.grid_velocity`. Writing it
`a(x, t)`:

```
                ┌──────────────────────────────────┐
                │  ∂φ/∂t|_x  =  ∂φ/∂t|_r  +  a φ_x │        (ALE identity)
                └──────────────────────────────────┘

        a(x, t) = (Ṙ/R) · (1 - x)^2 / (2L) · [1 + L (1 + x) / (1 - x)]
```

This holds for **any** field. There is no modelling content in it at all — no
constitutive assumption, no closure, no small parameter. It is the same identity
that turns a Eulerian derivative into a material derivative when the frame
moves; here the frame is the stretched grid attached to the wall.

### Equivalent reading: `a` is a drift speed

Hold a physical point fixed and ask how its computational coordinate changes.
From `r = R g(x) = const`:

```
0 = Ṙ g(x) + R g'(x) ẋ        ⟹      ∂x/∂t|_r = −(Ṙ/R) g(x) / g'(x) = −a
```

So `a = −∂x/∂t|_r`: the speed at which fixed physical points drift through the
computational grid, signed so that it acts as a transport velocity in the
right-hand-side form `φ_t = … + a φ_x`. When the bubble grows (`Ṙ > 0`), `a > 0`
and material drifts toward smaller `x` — correct, since a fixed radius is closer
to the wall in stretched coordinates once the wall has expanded.

## 3. Why it appears twice in the code

The state is the pair `(T, V)` with `V ≡ ∂T/∂t|_r` (note: the Eulerian time
derivative, *not* the nodal one). Apply the identity once per state variable:

| rows | equation | what the code adds |
| --- | --- | --- |
| `T` | `∂T/∂t\|_x = V + a T_x` — the definition of `V`, transformed | `A[T,T] += a ∂/∂x`, `A[T,V] = I` |
| `V` | `∂V/∂t\|_x = Y + a V_x`, where `Y` is the physics right-hand side | `A[V,V] += a ∂/∂x` |

Same identity, applied twice, **coefficient 1 each time**. In `matlab_legacy/`
these are `f_A_matrix.m` (the `T` rows) and `f_modify_A_bulk_map.m` (the `V`
rows); in the port they are `assemble_block` and `add_moving_grid_terms` in
`imrns/assemble/numpy_backend.py`.

## 4. Where the "fudge" impression comes from

The docstring of `matlab_legacy/common/f_modify_A_bulk_map.m` claims the code
implements

```
V_t = Y + (a_t − a a_x) T_x + 2a V_x − a^2 T_xx
```

but the code adds only `a V_x`. A reader comparing the two will reasonably
conclude the implementation is an unexplained simplification — a fudge.

It is not. **That docstring is correct mathematics for a different choice of
second state variable.** Take `W ≡ ∂T/∂t|_x` (the nodal derivative) instead of
`V ≡ ∂T/∂t|_r`. Then `V = W − a T_x`, and substituting into the `V` row above
and solving for `W_t` gives, exactly,

```
W_t = Y + (a_t − a a_x) T_x + 2a W_x − a^2 T_xx
```

The two are the same equation in different variables. This is verified
symbolically in `tests/test_moving_grid.py`: the difference between the two
forms reduces identically to zero once `W = ∂T/∂t|_x` is imposed.

So the history is almost certainly: the code once used `W = ∂T/∂t|_x`, with all
four terms; it was later reformulated in terms of `V = ∂T/∂t|_r`, where three of
them collapse; the docstring was never updated. The surviving hint is the
comment in `f_A_matrix.m`:

```matlab
% idxV now stores W = dT/dt|_r, not dT/dt|_x
```

The word "now" is the tell.

**Which formulation is better?** The implemented one, clearly. It needs one term
rather than four, avoids `a_t` and `a_x` (which would need either analytic
differentiation of the map or extra numerical derivatives), and avoids an
`a^2 T_xx` term that adds a spurious diffusion-like stencil with the wrong sign
structure under upwinding. The reformulation was a real simplification, not a
shortcut.

## 5. Numerical verification

All of these are in `tests/test_moving_grid.py`.

**`a` equals the drift speed.** Hold `r` fixed, perturb `R`, solve for `x`
numerically and difference. Agreement with `grid_velocity` to ~1e-9 across the
domain.

**The ALE identity holds, with nothing missing.** On a manufactured field
`φ̃(r, t)`, compare `∂φ/∂t|_x` against `∂φ/∂t|_r + a φ_x`. The residual is pure
`D1` truncation error and converges at second order:

| `xN` | relative residual | ratio |
| --- | --- | --- |
| 301 | 7.08e-03 | |
| 601 | 2.12e-03 | 3.35 |
| 1201 | 5.80e-04 | 3.65 |
| 2401 | 1.45e-04 | 4.00 |
| 4801 | 3.69e-05 | 3.93 |

A missing term would make this plateau at a nonzero value. It does not. Using
coefficient 2 instead of 1 gives a relative residual of 0.81 — i.e. the
coefficient is pinned, not merely plausible.

**It does the job it is there for.** Freeze a field in *physical* space
(`∂T/∂t|_r = 0`, a Gaussian bump at `r = 6`) and integrate `∂T/∂t|_x = a T_x`
while the wall moves by 13%. The bump should stay put in `r`:

| | max error (bump peak = 1.0) |
| --- | --- |
| with the term | 4.7e-05 |
| with the term removed (`a := 0`) | 2.8e-01 |

A factor of 6000. Without it the scheme pins the field to grid nodes that are
themselves sliding through physical space.

**Boundary behaviour is sane.** `a → 0` as `x → 1`, so there is no spurious
transport at the far field. At the wall `a(−1) = 2Ṙ/(RL)`, and
`sign(a) = sign(Ṙ)`.

## 6. Upwinding

`f_advective_D1_operator` / `imrns.operators.upwind_advection` splits on the
sign of `a`. For the right-hand-side form `φ_t = … + a φ_x`, characteristics
satisfy `dx/dt = −a`, so for `a > 0` information arrives from larger `x` and the
stencil must be **forward** biased; for `a < 0`, backward. That is what the code
does, and the comment there is correct.

One consequence worth knowing. During expansion (`Ṙ > 0`, `a > 0`) the wall end
`x = −1` is an *outflow* boundary and needs no data. During collapse (`a < 0`)
it becomes *inflow*, and the backward-biased operator falls back to a one-sided
stencil at the first node rather than using boundary data. This is standard
practice and is self-consistent here, because the physical wall condition is
imposed on the `V` row at that node (tangential traction) and `T` at the wall
then follows from the `T`-row equation. It is not a defect, but it is the first
place to look if near-wall behaviour during a violent collapse ever looks wrong.

## 7. Summary

| Question | Answer |
| --- | --- |
| Does the term belong? | Yes. Without it the solver integrates the wrong equation whenever `Ṙ ≠ 0`. |
| Is it a fudge? | No. It is the ALE chain rule, derivable in three lines, with no modelling content. |
| Is the coefficient right? | Yes — 1, pinned numerically; 2 is wrong by O(1). |
| Is the implementation right? | Yes, in both places it appears, and the upwind direction is correct. |
| So why does it look unmotivated? | A stale docstring describing a superseded choice of state variable. The math in it is correct, just for the other formulation. |
