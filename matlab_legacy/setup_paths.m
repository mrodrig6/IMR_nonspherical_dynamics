function paths = setup_paths(varargin)
%SETUP_PATHS Add repository folders, and optionally IMRv2, to the MATLAB path.
%
%   setup_paths adds common/, cmap/, and examples/ from this repository.
%
%   setup_paths('IMRv2Path', pathToForwardSolver) also adds an explicit IMRv2
%   forward-solver folder containing f_imr_fd.m.
%
%   If IMRv2Path is omitted, setup_paths checks the IMRV2_FORWARD_SOLVER
%   environment variable and then ../IMRv2/src/forward_solver.

parser = inputParser;
parser.FunctionName = 'setup_paths';
addParameter(parser, 'IMRv2Path', "", @(p) ischar(p) || isstring(p));
parse(parser, varargin{:});

repoRoot = fileparts(mfilename('fullpath'));
commonDir = fullfile(repoRoot, 'common');
cmapDir = fullfile(repoRoot, 'cmap');
examplesDir = fullfile(repoRoot, 'examples');

addpath(commonDir);
addpath(cmapDir);
addpath(examplesDir);

imrv2Dir = resolve_imrv2_path(repoRoot, parser.Results.IMRv2Path);
if strlength(imrv2Dir) > 0
    addpath(char(imrv2Dir));
end

paths = struct();
paths.repoRoot = repoRoot;
paths.commonDir = commonDir;
paths.cmapDir = cmapDir;
paths.examplesDir = examplesDir;
paths.imrv2ForwardSolverDir = char(imrv2Dir);
paths.hasIMRv2 = exist('f_imr_fd', 'file') == 2;

if nargout == 0
    fprintf('Added repository paths:\n');
    fprintf('  %s\n', commonDir);
    fprintf('  %s\n', cmapDir);
    fprintf('  %s\n', examplesDir);
    if paths.hasIMRv2
        fprintf('IMRv2 radial solver available: %s\n', which('f_imr_fd'));
    else
        fprintf('IMRv2 radial solver not found. Free examples can still run.\n');
    end
    clear paths
end
end

function imrv2Dir = resolve_imrv2_path(repoRoot, explicitPath)
explicitPath = string(explicitPath);
if strlength(explicitPath) > 0
    if exist(char(explicitPath), 'dir') ~= 7
        error('setup_paths:InvalidIMRv2Path', ...
            'IMRv2Path does not exist or is not a folder: %s', char(explicitPath));
    end
    imrv2Dir = explicitPath;
    return;
end

envPath = string(getenv('IMRV2_FORWARD_SOLVER'));
if strlength(envPath) > 0
    if exist(char(envPath), 'dir') ~= 7
        error('setup_paths:InvalidIMRv2EnvironmentPath', ...
            'IMRV2_FORWARD_SOLVER does not exist or is not a folder: %s', char(envPath));
    end
    imrv2Dir = envPath;
    return;
end

defaultPath = string(fullfile(fileparts(repoRoot), 'IMRv2', 'src', 'forward_solver'));
if exist(char(defaultPath), 'dir') == 7
    imrv2Dir = defaultPath;
else
    imrv2Dir = "";
end
end
