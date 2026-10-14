addpath(fileparts(mfilename('fullpath')));
cfg = example_config("free", "viscoelastic", "multimode");
result = run_nonspherical_example(cfg);
