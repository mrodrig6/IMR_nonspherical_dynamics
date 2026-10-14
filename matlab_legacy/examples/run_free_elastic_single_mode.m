addpath(fileparts(mfilename('fullpath')));
cfg = example_config("free", "elastic", "single");
result = run_nonspherical_example(cfg);
