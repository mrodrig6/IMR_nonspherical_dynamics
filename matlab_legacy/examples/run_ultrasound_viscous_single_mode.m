addpath(fileparts(mfilename('fullpath')));
cfg = example_config("ultrasound", "viscous", "single");
result = run_nonspherical_example(cfg);
