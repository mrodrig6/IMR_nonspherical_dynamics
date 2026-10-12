addpath(fileparts(mfilename('fullpath')));
cfg = example_config("ultrasound", "viscoelastic", "multimode");
result = run_nonspherical_example(cfg);
