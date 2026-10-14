addpath(fileparts(mfilename('fullpath')));
cfg = example_config("lic", "viscous", "single");
result = run_nonspherical_example(cfg);
