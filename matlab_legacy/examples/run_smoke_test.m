%RUN_SMOKE_TEST Quick self-contained check that does not require IMRv2.

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(repoRoot);
setup_paths;

cfg = example_config("free", "viscoelastic", "single");
cfg.xN = 16;
cfg.tsteps = 20;
cfg.makePlots = false;
cfg.makeSnapshot = false;
cfg.verbose = false;

result = run_nonspherical_example(cfg);

assert(isequal(size(result.ep), [1 cfg.tsteps]), 'Unexpected perturbation output size.');
assert(all(isfinite(result.ep(:))), 'Perturbation output contains non-finite values.');

cfgPros = cfg;
cfgPros.model = "Pros";
cfgPros.outputTag = "free_viscoelastic_single_pros";

prosResult = run_nonspherical_example(cfgPros);

assert(isequal(size(prosResult.ep), [1 cfgPros.tsteps]), 'Unexpected Pros perturbation output size.');
assert(all(isfinite(prosResult.ep(:))), 'Pros perturbation output contains non-finite values.');
assert(all(isfinite(prosResult.T(:))), 'Pros rotational field contains non-finite values.');

fprintf('Smoke test passed: free viscoelastic single-mode example returned ep size %s.\n', ...
    mat2str(size(result.ep)));
