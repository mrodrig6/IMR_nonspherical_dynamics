addpath(fileparts(mfilename('fullpath')));
cfg = example_config("ultrasound", "elastic", "multimode");
result = run_nonspherical_example(cfg);
