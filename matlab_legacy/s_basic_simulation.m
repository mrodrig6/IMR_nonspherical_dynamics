%% basic simulation input
clear all
clc
% close all

addpath common
addpath cmap


%
tic
% -------- Radial Solver ----------------------------------------%
Rmax = 150-6;
Req = Rmax/1.2;
mu =  0.000;
G = 10e3;
alph = 0;
sig = 0.056;
p_a = -50e3; f_a = 28e3;
rho = 1048;
p8 = 101325;
tcLIC = Rmax*sqrt(rho/p8);
tf_nd = 8;%%max(texp1)/tcLIC; 
tsteps = 1000; ultra = false;

t = linspace(0, tf_nd, tsteps);
[t, R, Rd, Rdd] = f_call_IMRv2(Rmax, Req, mu, G, alph, sig, p_a, f_a, tf_nd, tsteps, ultra);

figure
plot(t, R, '-')
%%

% -------- perturbation solver initial conditions ---------------%
% define grid in transformed domain
xN = 256;
% Mode numbers
n = [2 5 8 11];
% n = 5;
L = 5;
Lmax = Rmax/Req;

x = linspace(-1,1,xN);
r = Lmax*(1+L*(2./(1-x)-1));


N = n;
forcedep = 'F';
ep0 = [-0.115 0.115 -0.055 -0.065].*ones(size(N));
% ep0 = 0.1;
epd0 = zeros(size(N));
T0 = 0.0./r.*ones(length(N), xN);
Td0 = T0;
T0(end) = 0;

% Characteristic scales
Lc = Req;
rhoc = rho;
tc = sqrt(rhoc/p8)*Lc;
Uc = Lc/tc;
pc = rhoc*Uc^2;

Ca = pc/G;
Re = Lc*sqrt(rhoc*pc)/mu;
We = pc*Lc/(2*sig);
Oh = sqrt(We)/Re;
De = Ca/Re;
Ec = sqrt(We)/Ca;

% -------- Re-nondimensionalize to perturbation scalings ------- %
R = R.*Rmax/Req;
Rdd = Rdd.*Req/Rmax;
t = t.*tcLIC/tc;

% -------- prescribed perturbation history, if forcedep == 'T' -------- %
solverOptions = {};
if forcedep == 'T'
    forcedEpAngularFrequency = 2*pi;
    forcedEpPeriod = 2*pi/forcedEpAngularFrequency;

    forcedEpAmplitude = ep0(:).';
    if isscalar(forcedEpAmplitude) && numel(N) > 1
        forcedEpAmplitude = repmat(forcedEpAmplitude, 1, numel(N));
    end

    if numel(forcedEpAmplitude) ~= numel(N)
        error('forcedEpAmplitude must be scalar or have one value per mode in N.');
    end

    forcedEp = zeros(numel(N), numel(t));
    forcedEpd = zeros(numel(N), numel(t));
    for iMode = 1:numel(N)
        forcedEp(iMode, :) = forcedEpAmplitude(iMode)*sin(forcedEpAngularFrequency.*t);
        forcedEpd(iMode, :) = forcedEpAmplitude(iMode)*forcedEpAngularFrequency.* ...
            cos(forcedEpAngularFrequency.*t);
    end

    solverOptions = {'ForcedEp', forcedEp, 'ForcedEpd', forcedEpd, ...
        'ForcedEpPeriod', forcedEpPeriod};
end
solverOptions = {'Verbose', true};

mod = "me";
[ep, ~, ~, ~, ~, ~, ~] = compute_rotational_perturbation_evolution(xN, L, N, ep0, epd0, T0, ...
    Td0, 1, R, Rd, Rdd, Ca, alph, Re, We, t, 2, forcedep, mod, "irr", solverOptions{:});
epirr = ep;


[ep, epd, T, Td, R, Rd, t] = compute_rotational_perturbation_evolution(xN, L, N, ep0, epd0, T0, ...
    Td0, 1, R, Rd, Rdd, Ca, alph, Re, We, t, 2, forcedep, mod, "rot", solverOptions{:});

% [eppros, epdpros, ~, ~, ~, ~, ~] = compute_rotational_perturbation_evolution(xN, L, N, ep0, epd0, T0, ...
%     Td0, 1, R, Rd, Rdd, Ca, alph, Re, We, t, 2, forcedep, "Pros", "rot", solverOptions{:});
toc
%%

Lmax = Rmax/Req;
% figure
hold on
plot(t./Lmax, R./Lmax)

for i = 1:length(n)
figure(3)
hold on
plot(t./Lmax, ep(i,:))
hold on

% plot(t./Lmax, epirr(i, 1:length(t)), '--')
% plot(t./Lmax, eppros(i,:), '-.')
ylim([-.1 .1])
end
%%
make_axisym_displacement_movie_all_fields(T, ep, R, t, N, L, fullfile(pwd,'strain_prelim.pdf'), ...
    'Rref', 1, ...
    'StrainMeasure', 'almansi', ... 
    'StrainScalar', 'ert', ...
    'OutputMode', 'snapshot_pdf', ...
    'SnapshotLayout', [3 4], ...
    'SnapshotWidthNormalized', 0.75, ...
    'SnapshotHeightNormalized', 0.85, ...
    'SnapshotTimeRange', [0 t(end)], ...
    'GridCircles', 24, ...
    'GridRays', 96, ...
    'tc', 1, ...
    'RLimEq', [1 6], ...
    'FEM_grid', true, ...
    'GridCircleSpacing', 'current', ...
    'GridColor', [1 1 1], ...
    'StrainColormap', @brown_centered_diverging, 'StrainCLim', [-0.1 0.1], ...
    'SymmetricCLim', true,  'SnapshotTileSpacing', 'compact', ...
    'SnapshotPadding', 'compact', 'ColorbarLabel', 'Eulerian Almansi shear strain $e_{rt}$');


%%
clear make_axisym_displacement_movie_all_fields

make_axisym_displacement_movie_all_fields(T, ep, R, t, N, L, ...
    fullfile(pwd, 'strain_movie.mp4'), ...
    'Rref', 1, ...
    'OutputMode', 'movie', ...
    'MovieType', 'mp4', ...
    'WriteMovie', true, ...
    'FrameRate', 60, ...
    'MovieLength', 3, ...
    'ShowMovieColorbar', false, ...
    'StrainMeasure', 'almansi', ...
    'StrainScalar', 'ert', ...
    'StrainColormap', @brown_centered_diverging, ...
    'StrainCLim', [-0.2 0.2], ...
    'SymmetricCLim', true, ...
    'cbScale', 'linear', ...
    'FEM_grid', true, ...
    'GridType', 'eulerian', ...
    'GridCircles', 32, ...
    'RadialSpacingPower', 1, ...
    'GridRays', 36, ...
    'RLimEq', [1 5], ...
    'GridColor', [1 1 1], ...
    'MovieFigurePosition', [100 100 2160 2160]);
