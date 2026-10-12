# Porting notes

What changed moving from MATLAB to Python, and why. The original is preserved
unmodified in `matlab_legacy/`.

## Defects found in the original

Listed so that published results can be checked. None of these were introduced
by the port.

### Change results

**`Rmax = 150-6` in `s_basic_simulation.m:13`.** Almost certainly `150e-6`
(150 µm). As written the driver's bubble is 144 metres. Because
`Lc = Req = Rmax/1.2`, every dimensionless group computed in that driver is
wrong, not merely the length scale.

**`solverOptions` overwritten, `s_basic_simulation.m:75-100`.** The
`forcedep == 'T'` branch assembles `{'ForcedEp', ..., 'ForcedEpd', ...}`, and
line 100 then unconditionally replaces it with `{'Verbose', true}`. Prescribed-
amplitude runs from that driver could not work; the solver errors on the
missing histories.

**Two grid spacings on one grid.** `compute_rotational_perturbation_evolution.m`
builds `x_g = linspace(-1, 1-1e-4, xN)` and passes
`h = x_g(2)-x_g(1) = (2-1e-4)/(xN-1)` to `f_difference_stencil`, while
`f_A_matrix` and `f_modify_A_bulk_map` both hard-code `h = 2/(xN-1)` for the
advection operator. The two operator families therefore used different spacings
on the same grid, a relative inconsistency of 5e-5. The port derives one `h`
from the grid (`imrns.grid.grid_spacing`) and gives it to both.

**Truncated outputs desynchronised.** In the quasi-equilibrium branch,
`T_tot`, `Td_tot`, `ep_tot` and `epd_tot` are truncated at `i+1` but `t`, `R`
and `Rd` are returned at full length, so callers get
`numel(t) != size(ep, 2)`. The commented-out early-exit block immediately below
does truncate them, which suggests an oversight. In the port every array is
trimmed together and `PerturbationResult.termination` says why.

**`cmap/` was never committed.** `setup_paths` added it, the README documented
it, `s_basic_simulation.m` passed `@brown_centered_diverging`, and
`example_config` defaulted to it — but the directory is absent. Every example
with `makeSnapshot = true` failed. The solver also called `magma(...)`, which is
not a MATLAB builtin. Both are now real code in `imrns.colormaps`.

### Smaller

- `addpath(...)` ran on every call to the solver — a filesystem side effect in
  a hot function.
- Under `Verbose`, `condest` plus `rcond(full(Alhs))` costs about 220 ms,
  roughly 2.6 timesteps, and reported a stale matrix whenever the cache was
  active.
- `f_boundary_conditions`, `"Pros"` branch: `A(idxT(end), :) = 0` zeroes the row
  across *all* modes where the `"me"` branch correctly scopes to the block.
  Harmless today because there are no cross-mode entries, but fragile.
- `s_basic_simulation.m:50`: `T0(end) = 0` linear-indexes a matrix; presumably
  `T0(:,end) = 0`. Harmless only because `T0` is already zero.
- `check_quasi_equil` hardcodes `if ti < 25*P`.
- `f_boundary_conditions` comments read `fReqm` and `pReqpto` — a find/replace
  of `ro` to `Req` ran over the prose, in the file that defines the wall
  traction.
- `f_trapz_integration_matrices` filled `W` with an `O(xN^2)` double loop.

## Deliberately preserved

**The `f_ep_source_terms` / `f_modify_A_bulk_map` ordering.**
`f_ep_source_terms` captures `Li = A(idxV, whole_block)` to substitute the
second time derivative of `T`, and it runs *before* `f_modify_A_bulk_map` adds
the moving-grid advection to those same `V` rows. The substitution therefore
uses the operator without the mapping correction, which changes answers for
moving-radius runs.

Confirmed with the author as intentional. Preserved exactly, and commented as
load-bearing in both `assemble/numpy_backend.py` and `assemble/jax_backend.py`
so it is not "tidied up" later.

## Structural changes

**The operator is block diagonal by mode.** Every write the assembly performs
lands inside the writing mode's own block — the linearised modes do not couple.
The MATLAB code nevertheless built and factorised one `n_modes * block_size`
system. The port solves each mode separately, which for four modes at
`xN = 256` replaces a 2056-cubed factorisation with four 514-cubed ones.
`tests/test_physics_limits.py::test_modes_are_independent` pins the property
this relies on.

**Factorisations are cached, not matrices.** The MATLAB code detected a
constant operator and cached the *matrix*, then called `\` on it every step,
re-factorising each time. The port caches the LU factorisation.

**The solver no longer draws figures.** The quasi-equilibrium branch created a
`tiledlayout`, set LaTeX font sizes, drew a colorbar and returned from the
middle of the time loop. Early termination is now reported through
`PerturbationResult.termination`.

**Index bookkeeping is computed once.** `f_get_indicies` was recomputed inside
every assembly routine, once per mode per timestep. It is now a `StateLayout`
built once per run.

**Nondimensionalisation happens in one place.** The same five lines appeared in
`s_basic_simulation.m` and `run_nonspherical_example.m`, with the radial
history rescaled inline in both.

**Eighteen three-line scripts became eighteen parameter files** plus one runner
and a CLI.

**The MEX dependency is gone.** `scipy.special.hyp2f1` replaces
`mexhyp2f1`, removing two platform binaries, seven Cephes C sources, the
`mex -setup` requirement and the associated third-party licensing surface.

## Performance

Measured in this repository, four modes at `xN = 256`, 200 steps. The ordering
of these matters more than any individual number: the linear algebra had to be
restructured before compiling anything was worth doing.

| Change | ms/step |
| --- | --- |
| monolithic sparse solve, as originally assembled | 85 |
| per-mode block solves | 42 |
| slices instead of fancy indexing for sub-block writes | 31.5 |
| Numba kernels | 26.5 |
| constant radius, factorisation reused | 1.3 |

Where the time actually goes at `xN = 256`, per mode-block: about 6.5 ms to
assemble, 2.5 ms to factorise and solve. **Assembly dominates.** Numba makes
the isolated coefficient kernels about 50x faster (58.8 µs to 1.2 µs for `Ct7`)
but those kernels are a small share of assembly, so the end-to-end contribution
is about 20%. The bigger wins were avoiding `np.ix_` copies on 256x256
sub-blocks and hoisting the mode-independent advection operator out of the mode
loop.

### Measured and rejected

- **Banded-Schur elimination of the `T` block** (514-cubed down to 258-cubed):
  9x *slower*. The dense `V <- T` coupling makes the complement a wash
  asymptotically, and `solve_banded` with 258 right-hand sides is slow.
- **Previous step's LU as a GMRES preconditioner** for moving-radius runs: only
  about 1.5x at small drift between steps, and worse than refactorising at
  realistic drift.

## Verification strategy

Neither MATLAB nor Octave is used. The hand-expanded coefficients are the only
written statement of the model, so transcribing them is the main risk, and
`tests/matlab_oracle.py` removes transcription from the trust chain: it reads
`matlab_legacy/common/*.m`, extracts the right-hand side of a named assignment,
rewrites MATLAB array syntax as Python and evaluates it. Every kernel is
compared against its own source at `rtol=1e-13`.

That is a transcription check, not a correctness check, so it is backed by
independent physics: the Lamb capillary frequency, first-order convergence of
`rot` to `irr` as the deviatoric stress vanishes, exact linearity in the initial
amplitude, mode independence, and grid convergence. Backends are required to
agree to `1e-12`, gradients to match central differences to `2e-5`, and pinned
trajectories to stay fixed.

The one gap: this cannot prove agreement with specific saved MATLAB runs. If
`.mat` outputs from a known-good run are available they should be added as
regression fixtures.
