# Example Simulations

This folder contains ready-to-run MATLAB example scripts built from the workflow
in `../s_basic_simulation.m`.

The examples span:

- Radial drive type: `free`, `lic`, and `ultrasound`
- Material type: `viscous`, `elastic`, and `viscoelastic`
- Mode set: `single_mode` and `multimode`

Each `run_*.m` script creates a configuration with `example_config` and then
calls `run_nonspherical_example`. The scripts are not run automatically.

## Running One Example

From MATLAB:

```matlab
setup_paths
run('examples/run_lic_viscoelastic_single_mode.m')
```

or from inside this folder:

```matlab
run_lic_viscoelastic_single_mode
```

Generated figures and snapshot PDFs are written to `examples/output/` by
default.

For a quick check that does not require IMRv2, figures, or file output:

```matlab
run('examples/run_smoke_test.m')
```

## Drive Types

- `free`: uses a prescribed constant-radius equilibrium bubble history. This
  does not call the external IMR radial solver.
- `lic`: uses `f_call_IMRv2` with ultrasound disabled, matching the basic
  laser-induced cavitation style workflow in `s_basic_simulation.m`.
- `ultrasound`: uses `f_call_IMRv2` with acoustic forcing enabled.

## Materials

The material presets are intentionally simple starting points:

- `viscous`: finite viscosity with negligible elastic stiffness.
- `elastic`: finite elastic stiffness with very small viscosity.
- `viscoelastic`: finite viscosity and elastic stiffness, matching the default
  style of `s_basic_simulation.m`.

Tune the parameters in each script or in `example_config.m` for publication
quality studies.
