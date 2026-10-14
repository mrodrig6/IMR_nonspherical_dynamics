function [t, R, Rd, Rdd] = f_call_IMRv2(Rmax, Req, mu, G, alph, sig, p_a, f_a, tf_nd, tsteps, ultra, varargin)
%F_CALL_IMRV2 Run the external IMRv2 radial solver and return bubble history.
%
%   f_call_IMRv2(..., ultra) looks for f_imr_fd on the MATLAB path, in the
%   IMRV2_FORWARD_SOLVER environment variable, or in ../IMRv2/src/forward_solver.
%
%   f_call_IMRv2(..., ultra, 'IMRv2Path', pathToForwardSolver) adds an explicit
%   IMRv2 forward-solver folder before calling f_imr_fd.

opts = parse_imrv2_options(varargin{:});
imrv2Path = resolve_imrv2_forward_solver_path(varargin{:});
if strlength(imrv2Path) > 0
    addpath(char(imrv2Path));
end

if exist('f_imr_fd', 'file') ~= 2
    error('f_call_IMRv2:MissingIMRv2', ['Could not find f_imr_fd. Run setup_paths ' ...
        'with the IMRv2 forward-solver folder, set IMRV2_FORWARD_SOLVER, or add ' ...
        '../IMRv2/src/forward_solver to the MATLAB path.']);
end

% equation options
% ------- Material Properties ------------------------%
kappa = opts.Kappa;
T8 = 298.15;
rho8 = 1000;

% ------- Simulation settings ------------------------%
radial = 2;
% vapor = 1;
% collapse = 0;
% bubtherm = 1;
% medtherm = 0;
% masstrans = 1;
% stress = 2;
vapor = 0;
collapse = 0;
bubtherm = 0;
medtherm = 0;
masstrans = 0;
stress = 2;




if ultra
    % --------- Ultrasound settings -----------------------%
    pa = p_a;
    omega = 2*pi*f_a;
    wavetype = 4;
else
    pa = 0;
    omega = 0;
    wavetype = 0;
end

% ------ Simulation time ---------------------------- %
tc = Rmax*sqrt(rho8/101325);
tfin = tf_nd*tc;
tvector = linspace(0,tfin,tsteps);
varin = {'progdisplay',0,'radial',radial,'bubtherm',bubtherm,'tvector',tvector,...
         'vapor',vapor,'medtherm',medtherm,'masstrans',masstrans,'method',23,...
         'stress',stress,'collapse',collapse,'mu',mu,'g',G,'lambda1',0e-7,...
         'lambda2',0,'alphax', alph, 'surft', sig,'r0',Rmax,'req',Req,'kappa',kappa,'t8',T8,...
         'rho8',rho8, 'pa',pa, 'omega', omega, 'wave_type', wavetype};

[t,R,Rd,~,~,~,~,Rdd] = f_imr_fd(varin{:},'Nt',75);
end

function imrv2Path = resolve_imrv2_forward_solver_path(varargin)
parser = inputParser;
parser.FunctionName = 'f_call_IMRv2';
addParameter(parser, 'IMRv2Path', "", @(p) ischar(p) || isstring(p));
addParameter(parser, 'Kappa', 1, @(v) isnumeric(v) && isscalar(v) && isfinite(v));
parse(parser, varargin{:});

explicitPath = string(parser.Results.IMRv2Path);
if strlength(explicitPath) > 0
    if exist(char(explicitPath), 'dir') ~= 7
        error('f_call_IMRv2:InvalidIMRv2Path', ...
            'IMRv2Path does not exist or is not a folder: %s', char(explicitPath));
    end
    imrv2Path = explicitPath;
    return;
end

envPath = string(getenv('IMRV2_FORWARD_SOLVER'));
if strlength(envPath) > 0
    if exist(char(envPath), 'dir') ~= 7
        error('f_call_IMRv2:InvalidIMRv2EnvironmentPath', ...
            'IMRV2_FORWARD_SOLVER does not exist or is not a folder: %s', char(envPath));
    end
    imrv2Path = envPath;
    return;
end

repoRoot = fileparts(fileparts(mfilename('fullpath')));
defaultPath = string(fullfile(fileparts(repoRoot), 'IMRv2', 'src', 'forward_solver'));
if exist(char(defaultPath), 'dir') == 7
    imrv2Path = defaultPath;
else
    imrv2Path = "";
end
end

function opts = parse_imrv2_options(varargin)
parser = inputParser;
parser.FunctionName = 'f_call_IMRv2';
addParameter(parser, 'IMRv2Path', "", @(p) ischar(p) || isstring(p));
addParameter(parser, 'Kappa', 1, @(v) isnumeric(v) && isscalar(v) && isfinite(v));
parse(parser, varargin{:});
opts = parser.Results;
end
