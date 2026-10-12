addpath(fileparts(mfilename('fullpath')));
cfg = example_config("free", "viscous", "multimode");
result = run_nonspherical_example(cfg);
