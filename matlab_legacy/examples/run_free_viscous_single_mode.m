addpath(fileparts(mfilename('fullpath')));
cfg = example_config("free", "viscous", "single");
result = run_nonspherical_example(cfg);
