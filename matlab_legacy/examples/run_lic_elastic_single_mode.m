addpath(fileparts(mfilename('fullpath')));
cfg = example_config("lic", "elastic", "single");
result = run_nonspherical_example(cfg);
