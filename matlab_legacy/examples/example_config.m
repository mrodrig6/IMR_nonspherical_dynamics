function cfg = example_config(driveType, materialType, modeSet)
%EXAMPLE_CONFIG Create a baseline configuration for example simulations.
%
%   cfg = example_config(driveType, materialType, modeSet)
%
% driveType:
%   "free"        constant-radius free perturbation evolution
%   "lic"         laser-induced cavitation style radial history
%   "ultrasound"  acoustically forced radial history
%
% materialType:
%   "viscous", "elastic", or "viscoelastic"
%
% modeSet:
%   "single" or "multimode"

driveType = lower(string(driveType));
materialType = lower(string(materialType));
modeSet = lower(string(modeSet));

cfg = struct();
cfg.driveType = driveType;
cfg.materialType = materialType;
cfg.modeSet = modeSet;
cfg.outputTag = char(driveType + "_" + materialType + "_" + modeSet);

% Shared physical constants and surface tension.
cfg.rho = 1048;
cfg.p8 = 101325;
cfg.sig = 0.056;

% Shared perturbation solver settings.
cfg.xN = 256;
cfg.L = 5;
cfg.forcedep = 'F';
cfg.model = "me";
cfg.timeSteppingMethod = 2;

% Plot/output settings.
cfg.makePlots = true;
cfg.makeSnapshot = true;
cfg.verbose = false;
cfg.outputDir = "";
cfg.outputFile = "";
cfg.snapshotLayout = [3 4];
cfg.snapshotTimeRange = [];
cfg.strainScalar = 'ert';
cfg.strainColormap = 'brown_centered_diverging';
cfg.strainCLim = [-0.1 0.1];
cfg.gridColor = [0.24 0.28 0.31];
cfg.colorbarLabel = 'Eulerian Almansi shear strain $e_{rt}$';

% Default radial/acoustic settings. Drive-specific cases override these below.
cfg.Rmax = 150e-6;
cfg.Req = cfg.Rmax/3;
cfg.tf_nd = 4;
cfg.tsteps = 5000;
cfg.ultra = false;
cfg.p_a = -550e3;
cfg.f_a = 225e3;

switch driveType
    case "free"
        cfg.radialHistory = "free";
        cfg.Rmax = 50e-6;
        cfg.Req = 50e-6;
        cfg.tf_nd = 12;
        cfg.tsteps = 3000;
        cfg.ultra = false;
        cfg.description = "Free perturbation evolution about a constant-radius bubble.";

    case "lic"
        cfg.radialHistory = "imr";
        cfg.Rmax = 150e-6;
        cfg.Req = cfg.Rmax/3;
        cfg.tf_nd = 4;
        cfg.tsteps = 5000;
        cfg.ultra = false;
        cfg.description = "Laser-induced cavitation style collapse/growth history.";

    case "ultrasound"
        cfg.radialHistory = "imr";
        cfg.Rmax = 50e-6;
        cfg.Req = 50e-6;
        cfg.tf_nd = 20;
        cfg.tsteps = 5000;
        cfg.ultra = true;
        cfg.p_a = -550e3;
        cfg.f_a = 225e3;
        cfg.description = "Ultrasound-forced oscillation about the equilibrium radius.";

    otherwise
        error('Unknown driveType "%s". Use free, lic, or ultrasound.', driveType);
end

switch materialType
    case "viscous"
        cfg.mu = 0.015;
        cfg.G = 1e-9;
        cfg.alph = 0;
        cfg.materialDescription = "Viscous limit: finite viscosity, negligible elastic stiffness.";

    case "elastic"
        cfg.mu = 1e-6;
        cfg.G = 2.e3;
        cfg.alph = 0.1;
        cfg.materialDescription = "Nearly elastic limit: finite stiffness, very small viscosity.";

    case "viscoelastic"
        cfg.mu = 0.015;
        cfg.G = 2e3;
        cfg.alph = 0.1;
        cfg.materialDescription = "Viscoelastic material: finite viscosity and stiffness.";

    otherwise
        error('Unknown materialType "%s". Use viscous, elastic, or viscoelastic.', materialType);
end

switch modeSet
    case {"single", "single_mode"}
        cfg.N = 8;
        cfg.ep0 = 0.1;
        cfg.epd0 = 0;

    case {"multi", "multimode", "multi_mode"}
        cfg.N = [5 8 11];
        cfg.ep0 = [0.075 -0.085 0.05];
        cfg.epd0 = [0 0 0];

    otherwise
        error('Unknown modeSet "%s". Use single or multimode.', modeSet);
end

end
