addpath(fileparts(mfilename('fullpath')));
cfg = example_config("free", "viscoelastic", "single");
result = run_nonspherical_example(cfg);
