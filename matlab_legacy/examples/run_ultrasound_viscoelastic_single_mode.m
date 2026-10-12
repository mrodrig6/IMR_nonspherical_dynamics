addpath(fileparts(mfilename('fullpath')));
cfg = example_config("ultrasound", "viscoelastic", "single");
result = run_nonspherical_example(cfg);
