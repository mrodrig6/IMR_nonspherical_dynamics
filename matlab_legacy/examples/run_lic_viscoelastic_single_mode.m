addpath(fileparts(mfilename('fullpath')));
cfg = example_config("lic", "viscoelastic", "single");
result = run_nonspherical_example(cfg);
