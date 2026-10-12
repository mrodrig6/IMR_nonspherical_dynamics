addpath(fileparts(mfilename('fullpath')));
cfg = example_config("lic", "elastic", "multimode");
result = run_nonspherical_example(cfg);
