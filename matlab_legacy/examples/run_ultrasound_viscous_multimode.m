addpath(fileparts(mfilename('fullpath')));
cfg = example_config("ultrasound", "viscous", "multimode");
result = run_nonspherical_example(cfg);
