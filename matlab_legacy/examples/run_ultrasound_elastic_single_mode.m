addpath(fileparts(mfilename('fullpath')));
cfg = example_config("ultrasound", "elastic", "single");
result = run_nonspherical_example(cfg);
