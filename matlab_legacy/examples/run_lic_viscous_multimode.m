addpath(fileparts(mfilename('fullpath')));
cfg = example_config("lic", "viscous", "multimode");
result = run_nonspherical_example(cfg);
