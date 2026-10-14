function result = run_nonspherical_example(cfg)
%RUN_NONSPHERICAL_EXAMPLE Run one configured nonspherical IMR example.
%
% This follows the workflow in ../s_basic_simulation.m:
%   1. Build or prescribe a radial history.
%   2. Compute nondimensional material groups.
%   3. Run irrotational and rotational perturbation solvers.
%   4. Optionally plot histories and export a strain snapshot PDF.

if nargin < 1
    error('A configuration struct from example_config is required.');
end

exampleDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(exampleDir);
addpath(fullfile(repoRoot, 'common'));
addpath(fullfile(repoRoot, 'cmap'));

startDir = pwd;
cleanupObj = onCleanup(@() cd(startDir));
cd(repoRoot);

cfg = normalize_example_config(cfg, exampleDir);

N = cfg.N(:).';
nMode = numel(N);
ep0 = row_vector_for_modes(cfg.ep0, nMode, 'ep0');
epd0 = row_vector_for_modes(cfg.epd0, nMode, 'epd0');

fprintf('Running %s\n', cfg.outputTag);
fprintf('  Drive: %s\n', cfg.description);
fprintf('  Material: %s\n', cfg.materialDescription);
fprintf('  Modes: %s\n', mat2str(N));

[t, R, Rd, Rdd, Lmax] = build_radial_history(cfg);

% Characteristic scales, following s_basic_simulation.m.
Lc = cfg.Req;
rhoc = cfg.rho;
tc = sqrt(rhoc/cfg.p8)*Lc;
Uc = Lc/tc;
pc = rhoc*Uc^2;

if cfg.G == 0
    Ca = Inf;
else
    Ca = pc/cfg.G;
end

if cfg.mu == 0
    Re = Inf;
else
    Re = Lc*sqrt(rhoc*pc)/cfg.mu;
end

We = pc*Lc/(2*cfg.sig);
Oh = sqrt(We)/Re;
De = Ca/Re;
Ec = sqrt(We)/Ca;

T0 = zeros(nMode, cfg.xN);
Td0 = T0;

[epirr, ~, ~, ~, ~, ~, ~] = compute_rotational_perturbation_evolution( ...
    cfg.xN, cfg.L, N, ep0, epd0, T0, Td0, 1, R, Rd, Rdd, ...
    Ca, cfg.alph, Re, We, t, cfg.timeSteppingMethod, cfg.forcedep, cfg.model, "irr", ...
    'Verbose', cfg.verbose);

[ep, epd, T, Td, R, Rd, t] = compute_rotational_perturbation_evolution( ...
    cfg.xN, cfg.L, N, ep0, epd0, T0, Td0, 1, R, Rd, Rdd, ...
    Ca, cfg.alph, Re, We, t, cfg.timeSteppingMethod, cfg.forcedep, cfg.model, "rot", ...
    'Verbose', cfg.verbose);

if cfg.makePlots
    plot_example_histories(cfg, N, t, R, ep, epirr, Lmax);
end

if cfg.makeSnapshot
    if ~exist(cfg.outputDir, 'dir')
        mkdir(cfg.outputDir);
    end

    make_axisym_displacement_movie_all_fields(T, ep, R, t, N, cfg.L, cfg.outputFile, ...
        'Rref', 1, ...
        'StrainMeasure', 'almansi', ...
        'StrainScalar', cfg.strainScalar, ...
        'OutputMode', 'snapshot_pdf', ...
        'SnapshotLayout', cfg.snapshotLayout, ...
        'SnapshotWidthNormalized', 0.75, ...
        'SnapshotHeightNormalized', 0.85, ...
        'SnapshotTimeRange', cfg.snapshotTimeRange, ...
        'GridCircles', 24, ...
        'GridRays', 96, ...
        'tc', 1, ...
        'RLimEq', [1 6], ...
        'FEM_grid', true, ...
        'GridColor', cfg.gridColor, ...
        'StrainColormap', cfg.strainColormap, ...
        'StrainCLim', cfg.strainCLim, ...
        'SymmetricCLim', true, ...
        'SnapshotTileSpacing', 'compact', ...
        'SnapshotPadding', 'compact', ...
        'ColorbarLabel', cfg.colorbarLabel);
end

result = struct();
result.cfg = cfg;
result.t = t;
result.R = R;
result.Rd = Rd;
result.ep = ep;
result.epd = epd;
result.epirr = epirr;
result.T = T;
result.Td = Td;
result.dimensionless = struct('Ca', Ca, 'Re', Re, 'We', We, ...
    'Oh', Oh, 'De', De, 'Ec', Ec);

end

function cfg = normalize_example_config(cfg, exampleDir)
if ~isfield(cfg, 'makePlots')
    cfg.makePlots = true;
end
if ~isfield(cfg, 'makeSnapshot')
    cfg.makeSnapshot = true;
end
if ~isfield(cfg, 'verbose')
    cfg.verbose = false;
end
if ~isfield(cfg, 'outputDir') || strlength(string(cfg.outputDir)) == 0
    cfg.outputDir = fullfile(exampleDir, 'output');
end
if ~isfield(cfg, 'outputFile') || strlength(string(cfg.outputFile)) == 0
    cfg.outputFile = fullfile(cfg.outputDir, [cfg.outputTag '_strain.pdf']);
end
if ~isfield(cfg, 'snapshotTimeRange') || isempty(cfg.snapshotTimeRange)
    cfg.snapshotTimeRange = [];
end
if ~isfield(cfg, 'strainColormap') || isempty(cfg.strainColormap)
    cfg.strainColormap = 'brown_centered_diverging';
end
if ~isfield(cfg, 'gridColor') || isempty(cfg.gridColor)
    cfg.gridColor = [0.24 0.28 0.31];
end
if ~isfield(cfg, 'colorbarLabel') || isempty(cfg.colorbarLabel)
    cfg.colorbarLabel = 'Eulerian Almansi shear strain $e_{rt}$';
end
end

function values = row_vector_for_modes(values, nMode, name)
values = values(:).';
if isscalar(values) && nMode > 1
    values = repmat(values, 1, nMode);
end
if numel(values) ~= nMode
    error('%s must be scalar or have one value per mode.', name);
end
end

function [t, R, Rd, Rdd, Lmax] = build_radial_history(cfg)
switch string(cfg.radialHistory)
    case "free"
        t = linspace(0, cfg.tf_nd, cfg.tsteps);
        R = ones(size(t));
        Rd = zeros(size(t));
        Rdd = zeros(size(t));
        Lmax = 1;

    case "imr"
        tcLIC = cfg.Rmax*sqrt(cfg.rho/cfg.p8);
        [t, R, Rd, Rdd] = f_call_IMRv2(cfg.Rmax, cfg.Req, cfg.mu, cfg.G, ...
            cfg.alph, cfg.sig, cfg.p_a, cfg.f_a, cfg.tf_nd, cfg.tsteps, cfg.ultra);

        tc = sqrt(cfg.rho/cfg.p8)*cfg.Req;
        R = R.*cfg.Rmax/cfg.Req;
        Rdd = Rdd.*cfg.Req/cfg.Rmax;
        t = t.*tcLIC/tc;
        Lmax = cfg.Rmax/cfg.Req;

    otherwise
        error('Unknown radialHistory "%s".', cfg.radialHistory);
end
end

function plot_example_histories(cfg, N, t, R, ep, epirr, Lmax)
figure('Name', [cfg.outputTag ' radius'], 'Color', 'w');
plot(t./Lmax, R./Lmax, '-', 'LineWidth', 1.5);
grid on;
xlabel('t^*');
ylabel('R^*');
title(strrep(cfg.outputTag, '_', ' '));

figure('Name', [cfg.outputTag ' perturbations'], 'Color', 'w');
hold on;
plot(t./Lmax, ep.', '-', 'LineWidth', 1.5);
plot(t./Lmax, epirr.', '--', 'LineWidth', 1.2);
grid on;
xlabel('t^*');
ylabel('\epsilon_n');
ylim([-0.1 0.1]);

labels = cell(1, 2*numel(N));
for i = 1:numel(N)
    labels{i} = sprintf('rot n=%d', N(i));
    labels{i + numel(N)} = sprintf('irr n=%d', N(i));
end
legend(labels, 'Location', 'best');
title(strrep(cfg.outputTag, '_', ' '));
end
