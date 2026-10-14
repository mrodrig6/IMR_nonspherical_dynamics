addpath(fileparts(mfilename('fullpath')));
cfg = example_config("lic", "viscoelastic", "multimode");
result = run_nonspherical_example(cfg);
