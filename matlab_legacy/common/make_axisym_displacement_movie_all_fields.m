function make_axisym_displacement_movie_all_fields(T, ep, R, t, N, L, outputFile, varargin)
%MAKE_AXISYM_DISPLACEMENT_MOVIE_ALL_FIELDS
% Clean Eulerian/current-domain visualization of strain, strain rate, and stress
% around a nonspherical bubble.
%
% One-function version:
%   StrainMeasure = 'almansi'      -> Eulerian Almansi strain e_ij
%   StrainMeasure = 'small'        -> infinitesimal/symmetric displacement gradient
%   StrainMeasure = 'strain_rate'  -> rate of strain D_ij
%   StrainMeasure = 'stress'       -> Cauchy/constitutive stress sigma_ij
%
% Near-wall plotting convention:
%   The displayed wall is the perturbed interface r_b(theta,phi), but the
%   solved radial fields live outside the mean spherical wall r=R(t).  The
%   default WallFieldEvaluationMode='mean_wall_extension' evaluates the
%   continuum field at max(r,R(t)) for rows with r<R(t), while the colour
%   surface is still drawn down to the displayed perturbed wall.  This gives
%   a continuous-looking wall plot without evaluating the field in the
%   unsolved mean-wall interior.  Set MaskBelowMeanWall=true to hide those
%   plotting-only rows.
%
% This version deliberately separates two operations that were mixed in the
% older plotting functions:
%   (1) computing the Eulerian strain scalar at current spatial points, and
%   (2) drawing optional visual grid lines.
%
% The strain colour field is NOT drawn on a pushed-forward/deformed material
% mesh.  At every time step, the current radial grid used by the solver is
% rebuilt from f_r(x,R(t),L); T(r,t), Phi(r,t), and their radial derivatives
% are interpolated on that current grid; the Eulerian displacement-gradient
% form of the Almansi strain is evaluated pointwise on a smooth current-space
% plotting grid from the current bubble wall to the outer radius.
%
% Default angular reconstruction is axisymmetric m=0. Optional real sectoral
% m=n cross-sections are supported. The rotational correction used here is the
% solver's S=0 form u_rot = T_n(r,t)Y e_r - grad(Phi_n(r,t)Y); an independent
% curl[S_n(r,t)Y e_r] contribution from the full appendix maps is not included
% unless it has already been folded into the supplied T/Phi solution.
%
% Main scalar choices for strain:
%   'signed_maxabs_principal'  signed principal strain with largest magnitude
%   'max_principal'           largest principal strain
%   'err','ett','epp'          normal tensor components in spherical basis
%   'ert','erp','etp'          shear tensor components in spherical basis
%   'frobenius'                Frobenius-type magnitude of Almansi tensor
%
% Colour scaling:
%   cbScale = 'linear'         plot the requested signed/unsigned scalar directly
%   cbScale = 'log'            plot abs(scalar) with logarithmic color mapping
%
% To plot the Eulerian rate-of-strain tensor instead of strain, call with
%   'StrainMeasure','strain_rate'
% and select the component using either the existing strain component names
% ('err','ett','epp','ert','erp','etp') or D aliases
% ('drr','dtt','dpp','drt','drp','dtp'). For example:
%   'StrainMeasure','strain_rate','StrainScalar','ert'
% plots D_{r theta}. For accurate rate-of-strain plots, pass the solver
% time derivatives when available:
%   'Td',Td,'epd',epd,'Rd',Rd
% where Td = dT/dt at fixed current r, epd = dep/dt, and Rd = dR/dt.
% Aliases 'Tdot' or 'V', 'epdot', and 'Rdot' are also accepted. If these
% are omitted, the function falls back to finite differences in the saved
% output frames.
%%
% To plot stress, call with
%   'StrainMeasure','stress','StressScalar','srp','G',G,'mu',mu,'alpha',alpha
% For shear stress components pressure is irrelevant. For normal stress
% components, pass 'Pressure',P or 'P',P if you want the full Cauchy stress
% rather than the constitutive material part.
%
% Default near-wall behavior fixes artificial field values for r < R(t):
%   'WallFieldEvaluationMode','mean_wall_extension'
% The raw/legacy below-mean-wall field evaluation can be requested with:
%   'WallFieldEvaluationMode','perturbed_wall','MaskBelowMeanWall',false

% The optional material/eulerian grids are wall-to-domain remeshes of the
% pushed-forward total solution. They are visual overlays only and are not
% used to compute the strain colour field. Use GridType='pushed_forward_raw'
% to display literal material trajectories without radial remeshing.

    p = inputParser;
    % Exact name matching is required so short physical names such as 'G'
    % are not confused with GridCircles/GridColor/GridRays/GridType.
    p.PartialMatching = false;

    % Base geometry and output options.
    p.addParameter('Rref', []);
    p.addParameter('R0', []);
    p.addParameter('Req', []);              % legacy dimensional scale; solver coordinates use Req=1
    p.addParameter('EqFrameIdx', []);
    p.addParameter('RLimEq', []);
    p.addParameter('OutputMode', 'auto');      % 'auto','snapshot_pdf','movie'
    p.addParameter('SnapshotLayout', []);
    p.addParameter('SnapshotFrameIdx', []);
    p.addParameter('SnapshotTimeRange', []);
    p.addParameter('SnapshotWidthNormalized', 0.90);
    p.addParameter('SnapshotHeightNormalized', 0.90);
    p.addParameter('SnapshotFigureUnits', 'pixels');
    p.addParameter('SnapshotFigurePosition', []);
    p.addParameter('SnapshotTileSpacing', 'compact');
    p.addParameter('SnapshotPadding', 'compact');
    p.addParameter('SnapshotTiledLayoutPosition', []);
    p.addParameter('SnapshotAxesPosition', []);
    p.addParameter('SnapshotDPI', 300);
    p.addParameter('SnapshotAxes', []);
    p.addParameter('ShowSnapshotColorbar', true);
    p.addParameter('SnapshotColorbarPosition', []);
    p.addParameter('SnapshotColorbarFontSize', 16);
    p.addParameter('SnapshotColorbarLabelFontSize', 16);
    p.addParameter('ColorbarFontSize', []);
    p.addParameter('ColorbarLabelFontSize', []);
    p.addParameter('SnapshotPanelLabelStart', 1);
    p.addParameter('ShowPanelLabels', true);
    p.addParameter('PanelLabelFontSize', 16);
    p.addParameter('PanelLabelYOffset', 0.02);
    p.addParameter('MovieType', 'auto');       % 'auto','gif','mp4'
    p.addParameter('FrameIdx', []);
    p.addParameter('MovieTimeRange', []);
    p.addParameter('FrameRate', 15);
    p.addParameter('MovieLength', []);
    p.addParameter('DelayTime', []);
    p.addParameter('MovieFigurePosition', [100 100 1400 1400]);
    p.addParameter('MP4Profile', 'MPEG-4');
    p.addParameter('MP4Quality', 95);
    p.addParameter('WriteMovie', true);
    p.addParameter('ShowMovieColorbar', true);
    p.addParameter('Verbose', true);

    % Plot appearance.
    p.addParameter('FigureColor', [1 1 1]);
    p.addParameter('ShowTitle', true);
    p.addParameter('ShowAxesBox', true);
    p.addParameter('TitleFontSize', 16);
    p.addParameter('tc', []);
    p.addParameter('AxesAspect', 'equal');     % 'equal' or 'fill'
    p.addParameter('DisplayXLim', []);
    p.addParameter('DisplayZLim', []);
    p.addParameter('SquareFillFrame', true);
    p.addParameter('SquareFillFactor', sqrt(2));
    p.addParameter('BubbleWidth', 1.9);
    p.addParameter('BubbleColor', [0 0 0]);
    p.addParameter('FillBubble', true);
    p.addParameter('BubbleFillColor', [1 1 1]);
    p.addParameter('BubbleFillInsetFrac', 0.006); % redraw white interior slightly inside wall after field
    p.addParameter('ThetaBubble', 900);

    % Strain field options.
    p.addParameter('ColorMaterialByStrain', true);
    p.addParameter('StrainMeasure', 'almansi');
    % StrainMeasure = 'almansi' or 'small' preserves the old behaviour.
    % StrainMeasure = 'strain_rate'/'rate' plots D = 1/2*(grad(v)+grad(v)^T).
    p.addParameter('StrainScalar', 'signed_maxabs_principal');
    p.addParameter('StrainRateScalar', []);   % optional override when StrainMeasure is strain_rate

    % Optional time derivatives for accurate rate-of-strain plots.
    % Td/Tdot/V must have the same size as T: [numel(N) x xN x Nt].
    % epd/epdot must have the same size as ep: [numel(N) x Nt].
    % Rd/Rdot must have the same size as R: [Nt x 1] or [1 x Nt].
    p.addParameter('Td', []);
    p.addParameter('Tdot', []);
    p.addParameter('V', []);
    p.addParameter('epd', []);
    p.addParameter('epdot', []);
    p.addParameter('Rd', []);
    p.addParameter('Rdot', []);
    p.addParameter('UseFiniteDifferenceForRate', true);

    % Stress-field options.  For StrainMeasure='stress', the plotted
    % quantity is sigma = G*(1 + alpha*(I_B - 3))*B + 2*mu*D - P*I.
    % If Pressure/P is omitted, normal components are the constitutive
    % material part without the unknown hydrostatic pressure; shear components
    % are unaffected by pressure.
    p.addParameter('StressScalar', []);      % optional override for stress components
    p.addParameter('G', []);                 % shear modulus
    p.addParameter('mu', []);                % viscosity
    p.addParameter('alpha', 0);              % strain-stiffening parameter
    p.addParameter('ShearModulus', []);      % alias for G
    p.addParameter('Viscosity', []);         % alias for mu
    p.addParameter('StrainStiffeningAlpha', []); % alias for alpha
    p.addParameter('Pressure', []);          % optional pressure, scalar or time vector
    p.addParameter('P', []);                 % alias for Pressure

    p.addParameter('StrainColormap', parula(256));
    p.addParameter('StrainCLim', []);
    p.addParameter('cbScale', 'linear');       % 'linear' or 'log'; log plots abs(quantity)
    p.addParameter('StrainPercentile', 98);
    p.addParameter('SymmetricCLim', true);
    p.addParameter('ColorbarLabel', 'Eulerian Almansi strain');
    p.addParameter('ColorbarLocation', 'eastoutside');
    p.addParameter('StrainAlpha', 1.0);
    p.addParameter('StrainMeshR', 240);
    p.addParameter('StrainDisplayPhi', 721);
    p.addParameter('RadialSpacingPower', 1.05);
    p.addParameter('ThetaPoleEps', 1e-3);
    p.addParameter('RadialInterp', 'pchip');      % 'linear','pchip','makima'
    p.addParameter('ClampQueryBelowMeanWall', true);    % radial helper clamp; WallFieldEvaluationMode usually prevents below-wall queries
    p.addParameter('WallFieldEvaluationMode', 'mean_wall_extension'); % 'mean_wall_extension' evaluates field at max(r,R(t)); 'perturbed_wall' is legacy/raw
    p.addParameter('MaskBelowMeanWall', false);         % optionally hide color mesh where the display radius is below the solved mean wall
    p.addParameter('FillMissingColorData', true);
    p.addParameter('ClampColorDataToCLim', true);
    p.addParameter('ContourLevels', 96);
    p.addParameter('UseContourf', false);         % surface is smoother for dense grids

    % Angular reconstruction options.
    p.addParameter('AngularMode', 'axisymmetric');   % 'axisymmetric' or 'm_equals_n'
    p.addParameter('RealHarmonicPart', 'cos');       % 'cos' or 'sin' for m_equals_n
    p.addParameter('SlicePhi', 0);                   % azimuthal slice angle, radians
    p.addParameter('SlicePlane', 'auto');             % 'auto','meridional','equatorial'

    % Optional grid overlay. This is not used for the colour computation.
    p.addParameter('FEM_grid', false);
    p.addParameter('GridType', 'material');       % 'material'/'eulerian' = remeshed total-solution grid; 'pushed_forward_raw' = literal material map
    p.addParameter('GridCircles', 12);
    p.addParameter('GridRays', 48);
    p.addParameter('LineWidth', 0.65);
    p.addParameter('GridColor', [0.24 0.28 0.31]);
    p.addParameter('GridCircleSpacing', 'current'); % radial levels for GridType='material': 'current' or 'material'
    p.addParameter('HideWallCircle', false);
    p.addParameter('WallCircleOffsetFrac', 0.012);
    p.addParameter('PtsPerCircle', 500);
    p.addParameter('PtsPerRay', 500);
    p.addParameter('RoGridVals', []);
    % Preserve closed rings when the first-order map places a near-wall
    % sample inside the finite-amplitude level-set surface by O(ep^2).
    p.addParameter('KeepGridCirclesOutsideBubble', true);
    p.addParameter('KeepGridRaysOutsideBubble', true);
    p.addParameter('GridCircleWallClearanceFrac', 5e-4);
    p.addParameter('GridPathSamples', []);        % sample count used to remesh total-solution grid paths
    p.addParameter('ThetaGridEq', []);
    p.addParameter('AngleChoice', 'reference');
    p.addParameter('AngleIterations', 2);
    p.addParameter('ClipInsideBubble', true);
    p.addParameter('AnchorRaysToBubble', true);

    p.parse(varargin{:});
    opt = p.Results;
    if isempty(opt.ColorbarFontSize)
        opt.ColorbarFontSize = opt.SnapshotColorbarFontSize;
    end
    if isempty(opt.ColorbarLabelFontSize)
        opt.ColorbarLabelFontSize = opt.SnapshotColorbarLabelFontSize;
    end
    opt.cbScale = normalize_color_scale(opt.cbScale);

    opt = normalize_rate_derivative_options(opt, T, ep, R);

    if is_stress_requested(opt)
        if strcmp(char(opt.ColorbarLabel), 'Eulerian Almansi strain')
            activeScalar = active_tensor_scalar(opt);
            opt.ColorbarLabel = default_stress_label(activeScalar);
        end
    elseif is_rate_of_strain_requested(opt)
        if strcmp(char(opt.ColorbarLabel), 'Eulerian Almansi strain')
            activeScalar = active_tensor_scalar(opt);
            opt.ColorbarLabel = default_rate_label(activeScalar);
        end
        % Keep colour limits symmetric by default for signed tensor components.
        % Users can still override with 'SymmetricCLim',false or explicit 'StrainCLim'.
    end

    R = R(:);
    t = t(:);
    N = N(:).';

    nModes = numel(N);
    xN = size(T,2);
    Nt = size(T,3);

    if size(T,1) ~= nModes
        error('size(T,1) must equal numel(N).');
    end
    if size(ep,1) ~= nModes || size(ep,2) ~= Nt
        error('ep must be size [numel(N) x size(T,3)].');
    end
    if numel(R) ~= Nt || numel(t) ~= Nt
        error('R and t must have length size(T,3).');
    end

    opt = normalize_stress_options(opt, Nt);

    if isempty(opt.EqFrameIdx)
        opt.EqFrameIdx = Nt;
    end
    itEq = round(opt.EqFrameIdx);
    itEq = max(1, min(Nt, itEq));

    if isempty(opt.Rref)
        if ~isempty(opt.R0)
            Rref = opt.R0;
        elseif ~isempty(opt.Req)
            % Repository workflows nondimensionalize R, ep, and T by Req
            % before calling this renderer (the perturbation solver is also
            % called with Req=1). Retain the historical 'Req' argument as a
            % compatibility signal without mixing its dimensional value into
            % the order-one material coordinate map. Same-unit callers can
            % set an explicit 'Rref' instead.
            validateattributes(opt.Req, {'numeric'}, {'scalar','real','positive','finite'});
            Rref = 1;
        else
            Rref = R(itEq);
        end
    else
        Rref = opt.Rref;
    end
    validateattributes(Rref, {'numeric'}, {'scalar','real','positive','finite'});

    renderIntoExistingAxes = ~isempty(opt.SnapshotAxes);
    [outDir, outBase, outExt] = fileparts(outputFile);
    extNoDot = lower(regexprep(outExt, '^\.', ''));
    outputMode = lower(char(opt.OutputMode));
    if isempty(outputMode) || strcmp(outputMode, 'auto')
        if renderIntoExistingAxes || ~isempty(opt.SnapshotLayout) || strcmp(extNoDot, 'pdf')
            outputMode = 'snapshot_pdf';
        else
            outputMode = 'movie';
        end
    end
    if ~ismember(outputMode, {'movie','snapshot_pdf'})
        error('OutputMode must be ''auto'', ''movie'', or ''snapshot_pdf''.');
    end

    if strcmp(outputMode, 'snapshot_pdf')
        validSnapshotExt = {'pdf','jpg','jpeg','png','tif','tiff'};
        if isempty(outExt)
            outputFile = fullfile(outDir, [outBase '.pdf']);
        elseif ~ismember(extNoDot, validSnapshotExt)
            error('Snapshot output must use .pdf, .jpg, .jpeg, .png, .tif, or .tiff.');
        end
        if ~renderIntoExistingAxes && exist(outputFile, 'file') == 2
            delete(outputFile);
        end
    else
        movieType = lower(char(opt.MovieType));
        if isempty(movieType) || strcmp(movieType, 'auto')
            if ~isempty(outExt)
                movieType = extNoDot;
            else
                movieType = 'gif';
            end
        end
        if strcmp(movieType, 'mpeg4')
            movieType = 'mp4';
        end
        if ~ismember(movieType, {'gif','mp4'})
            error('MovieType must be ''auto'', ''gif'', or ''mp4''.');
        end
        if isempty(outExt)
            outputFile = fullfile(outDir, [outBase '.' movieType]);
        end
        if opt.WriteMovie && exist(outputFile, 'file') == 2
            delete(outputFile);
        end
    end

    % Resolve snapshot frames.
    if isempty(opt.SnapshotLayout)
        snapshotLayout = [1 1];
    else
        snapshotLayout = round(opt.SnapshotLayout(:).');
        if numel(snapshotLayout) ~= 2 || any(snapshotLayout < 1)
            error('SnapshotLayout must be [nRows nCols].');
        end
    end
    nSnapshotTiles = prod(snapshotLayout);

    if isempty(opt.SnapshotFrameIdx)
        if isempty(opt.SnapshotTimeRange)
            snapStart = 1;
            snapEnd = Nt;
        else
            snapStart = nearest_time_index(t, opt.SnapshotTimeRange(1));
            snapEnd   = nearest_time_index(t, opt.SnapshotTimeRange(2));
            snapStart = max(1, min(Nt, snapStart));
            snapEnd   = max(snapStart, min(Nt, snapEnd));
        end
        if nSnapshotTiles == 1
            snapshotFrameIdx = round((snapStart + snapEnd)/2);
        else
            snapshotFrameIdx = round(linspace(snapStart, snapEnd, nSnapshotTiles));
        end
    else
        snapshotFrameIdx = round(opt.SnapshotFrameIdx(:).');
        snapshotFrameIdx = snapshotFrameIdx(snapshotFrameIdx >= 1 & snapshotFrameIdx <= Nt);
        if isempty(snapshotFrameIdx)
            error('SnapshotFrameIdx did not contain valid indices.');
        end
        if numel(snapshotFrameIdx) > nSnapshotTiles
            snapshotFrameIdx = snapshotFrameIdx(round(linspace(1, numel(snapshotFrameIdx), nSnapshotTiles)));
        end
    end

    % Resolve movie frames.
    movieTimeRange = opt.MovieTimeRange;

    % Convenience fallback: if making a movie and MovieTimeRange was not set,
    % allow SnapshotTimeRange to act as the movie time range too.
    if isempty(movieTimeRange) && ~isempty(opt.SnapshotTimeRange)
        movieTimeRange = opt.SnapshotTimeRange;
    end

    if isempty(movieTimeRange)
        movieStart = 1;
        movieEnd = Nt;
    else
        movieStart = nearest_time_index(t, movieTimeRange(1));
        movieEnd   = nearest_time_index(t, movieTimeRange(2));
        movieStart = max(1, min(Nt, movieStart));
        movieEnd   = max(movieStart, min(Nt, movieEnd));
    end
    if isempty(opt.FrameIdx)
        if isempty(opt.MovieLength)
            nMovieFrames = min(80, max(2, movieEnd - movieStart + 1));
        else
            nMovieFrames = max(2, round(opt.FrameRate * opt.MovieLength));
        end
        movieFrameIdx = round(linspace(movieStart, movieEnd, nMovieFrames));
    else
        movieFrameIdx = round(opt.FrameIdx(:).');
        movieFrameIdx = movieFrameIdx(movieFrameIdx >= 1 & movieFrameIdx <= Nt);
    end
    if strcmp(outputMode, 'snapshot_pdf')
        movieFrameIdx = [];
    end

    % Build quadrature/grid objects used by the existing model kernels.
    epsMap = 1e-12;
    xg = linspace(-1, 1-epsMap, xN).';
    [W, w, One_wT] = f_trapz_integration_matrices(xg);

    if strcmp(outputMode, 'snapshot_pdf')
        neededFrames = unique([snapshotFrameIdx, itEq], 'stable');
    else
        neededFrames = unique([movieFrameIdx, itEq], 'stable');
    end

    frameData = cell(Nt,1);
    for kk = 1:numel(neededFrames)
        it = neededFrames(kk);
        frameData{it} = build_frame_modal_data_current(it, T, ep, R, t, N, L, xg, W, w, One_wT, opt);
    end

    % Display outer radius.
    if isempty(opt.RLimEq)
        rOuterDisplay = min(max(frameData{neededFrames(1)}.rEval), 3*max(Rref, R(itEq)));
    else
        validateattributes(opt.RLimEq, {'numeric'}, {'vector','numel',2,'increasing','finite'});
        rOuterDisplay = opt.RLimEq(2);
    end

    if isempty(opt.DisplayXLim) || isempty(opt.DisplayZLim)
        if opt.SquareFillFrame
            halfSpan = rOuterDisplay / max(opt.SquareFillFactor, 1);
            opt.DisplayXLim = [-halfSpan, halfSpan];
            opt.DisplayZLim = [-halfSpan, halfSpan];
        else
            pad = 0.02*rOuterDisplay;
            opt.DisplayXLim = [-rOuterDisplay-pad, rOuterDisplay+pad];
            opt.DisplayZLim = [-rOuterDisplay-pad, rOuterDisplay+pad];
        end
    end

    % Colour limits.
    cMap = resolve_colormap(opt.StrainColormap);
    if opt.ColorMaterialByStrain
        if isempty(opt.StrainCLim)
            finiteAll = [];
            if strcmp(outputMode, 'snapshot_pdf')
                scanIdx = snapshotFrameIdx;
            else
                scanIdx = movieFrameIdx;
            end
            scanIdx = unique(scanIdx, 'stable');
            for kk = 1:numel(scanIdx)
                it = scanIdx(kk);
                if isempty(frameData{it})
                    frameData{it} = build_frame_modal_data_current(it, T, ep, R, t, N, L, xg, W, w, One_wT, opt);
                end
                [~,~,Ctmp] = current_almansi_colour_surface(frameData{it}, ep(:,it), R(it), Rref, N, rOuterDisplay, opt);
                vv = color_values_for_limits(Ctmp, opt);
                finiteAll = [finiteAll; vv(:)]; %#ok<AGROW>
            end
            if isempty(finiteAll)
                cLim = default_color_limits(opt);
            elseif is_log_color_scale(opt)
                cLo = prctile(finiteAll, 100 - opt.StrainPercentile);
                cHi = prctile(finiteAll, opt.StrainPercentile);
                if ~isfinite(cLo) || ~isfinite(cHi) || cHi <= cLo || cLo <= 0
                    cLo = min(finiteAll);
                    cHi = max(finiteAll);
                end
                if ~isfinite(cLo) || ~isfinite(cHi) || cHi <= cLo || cLo <= 0
                    cLim = default_color_limits(opt);
                else
                    cLim = [cLo cHi];
                end
            elseif opt.SymmetricCLim
                cHi = prctile(abs(finiteAll), opt.StrainPercentile);
                if ~isfinite(cHi) || cHi <= 0
                    cHi = max(abs(finiteAll));
                end
                if ~isfinite(cHi) || cHi <= 0
                    cHi = 1;
                end
                cLim = [-cHi cHi];
            else
                cLo = prctile(finiteAll, 100 - opt.StrainPercentile);
                cHi = prctile(finiteAll, opt.StrainPercentile);
                if ~isfinite(cLo) || ~isfinite(cHi) || cHi <= cLo
                    cLo = min(finiteAll);
                    cHi = max(finiteAll);
                end
                if cHi <= cLo
                    cLim = [-1 1];
                else
                    cLim = [cLo cHi];
                end
            end
        else
            cLim = normalize_explicit_color_limits(opt.StrainCLim, opt);
        end
    else
        cLim = [0 1];
    end

    if strcmp(outputMode, 'snapshot_pdf')
        render_snapshot_pdf(outputFile, snapshotFrameIdx, snapshotLayout, frameData, ep, R, N, t, Rref, rOuterDisplay, cLim, cMap, opt);
        return;
    end

    render_movie(outputFile, movieFrameIdx, frameData, ep, R, N, t, Rref, rOuterDisplay, cLim, cMap, opt, movieType);
end

function render_snapshot_pdf(outputFile, frameIdx, layout, frameData, ep, R, N, t, Rref, rOuterDisplay, cLim, cMap, opt)
    axList = opt.SnapshotAxes;
    renderIntoExistingAxes = ~isempty(axList);
    renderIntoDirectAxes = false;
    if renderIntoExistingAxes
        axList = axList(isgraphics(axList, 'axes'));
        nTiles = numel(axList);
        if nTiles ~= prod(layout)
            error('SnapshotAxes must contain prod(SnapshotLayout) axes handles.');
        end
        fig = ancestor(axList(1), 'figure');
    else
        if isempty(opt.SnapshotFigurePosition)
            fig = figure('Color', opt.FigureColor, 'Units', 'normalized', ...
                'Position', centered_normalized_position(opt.SnapshotWidthNormalized, opt.SnapshotHeightNormalized));
        else
            fig = figure('Color', opt.FigureColor, 'Units', opt.SnapshotFigureUnits, ...
                'Position', opt.SnapshotFigurePosition);
        end
        nTiles = prod(layout);
        if nTiles == 1 && ~isempty(opt.SnapshotAxesPosition)
            renderIntoDirectAxes = true;
            axList = axes(fig, 'Units', 'normalized', 'Position', opt.SnapshotAxesPosition); %#ok<LAXES>
        else
            tl = tiledlayout(fig, layout(1), layout(2), 'TileSpacing', opt.SnapshotTileSpacing, 'Padding', opt.SnapshotPadding);
            if isempty(opt.SnapshotTiledLayoutPosition)
                tl.Position = [0.04 0.04 0.84 0.92];
            else
                tl.Position = opt.SnapshotTiledLayoutPosition;
            end
        end
    end

    for kk = 1:nTiles
        if renderIntoExistingAxes || renderIntoDirectAxes
            ax = axList(kk);
        else
            ax = nexttile(tl, kk);
        end
        if kk <= numel(frameIdx)
            it = frameIdx(kk);
            render_one_tile(ax, frameData{it}, ep(:,it), R(it), N, t(it), Rref, rOuterDisplay, cLim, cMap, opt);
            add_snapshot_panel_label(ax, opt.SnapshotPanelLabelStart + kk - 1, opt);
        else
            axis(ax, 'off');
        end
    end

    if opt.ColorMaterialByStrain && opt.ShowSnapshotColorbar
        cbax = axes(fig, 'Position', [0.885 0.12 0.001 0.76], 'Visible', 'off'); %#ok<LAXES>
        colormap(cbax, cMap);
        caxis(cbax, cLim);
        apply_color_axis_scale(cbax, opt);
        cb = colorbar(cbax, 'eastoutside');
        if isempty(opt.SnapshotColorbarPosition)
            if renderIntoExistingAxes
                cb.Position = snapshot_row_colorbar_position(axList);
            else
                cb.Position = [0.90 0.12 0.02 0.76];
            end
        else
            cb.Position = opt.SnapshotColorbarPosition;
        end
        cb.TickLabelInterpreter = 'latex';
        cb.Label.Interpreter = 'latex';
        cb.Label.String = sanitize_latex_label(opt.ColorbarLabel);
        cb.Label.Rotation = 270;
        cb.FontSize = opt.SnapshotColorbarFontSize;
        cb.Label.FontSize = opt.SnapshotColorbarLabelFontSize;
    end

    if ~renderIntoExistingAxes
        [~,~,outExt] = fileparts(outputFile);
        if strcmpi(outExt, '.pdf')
            exportgraphics(fig, outputFile, 'ContentType', 'image', 'Resolution', opt.SnapshotDPI);
        elseif ismember(lower(regexprep(outExt, '^\.', '')), {'jpg','jpeg'})
            set(fig, 'InvertHardcopy', 'off', 'PaperPositionMode', 'auto');
            print(fig, outputFile, '-djpeg', sprintf('-r%d', round(opt.SnapshotDPI)));
        else
            exportgraphics(fig, outputFile, 'Resolution', opt.SnapshotDPI);
        end
    end
end

function pos = snapshot_row_colorbar_position(axList)
    drawnow;
    axList = axList(isgraphics(axList, 'axes'));
    axPos = reshape([axList.Position], 4, []).';
    rowBottom = min(axPos(:,2));
    rowTop = max(axPos(:,2) + axPos(:,4));
    rowCenter = 0.5 * (rowBottom + rowTop);
    rowHeight = rowTop - rowBottom;
    axRight = max(axPos(:,1) + axPos(:,3));

    cbHeight = min(0.90 * rowHeight, 0.30);
    cbWidth = 0.018;
    cbGap = 0.018;
    cbX = min(axRight + cbGap, 0.84);
    cbY = max(0.02, rowCenter - 0.5 * cbHeight);
    pos = [cbX cbY cbWidth cbHeight];
end

function add_snapshot_panel_label(ax, panelIdx, opt)
    if ~opt.ShowPanelLabels
        return;
    end
    yOffset = max(0, real(opt.PanelLabelYOffset));
    text(ax, 0, 1 + yOffset, panel_label_text(panelIdx), ...
        'Units', 'normalized', ...
        'Clipping', 'off', ...
        'Interpreter', 'latex', ...
        'FontSize', opt.PanelLabelFontSize, ...
        'FontWeight', 'bold', ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'bottom');
end

function label = panel_label_text(panelIdx)
    n = max(1, round(panelIdx));
    letters = '';
    while n > 0
        remIdx = mod(n - 1, 26);
        letters = [char('a' + remIdx) letters]; %#ok<AGROW>
        n = floor((n - 1) / 26);
    end
    label = [letters ')'];
end

function render_movie(outputFile, frameIdx, frameData, ep, R, N, t, Rref, rOuterDisplay, cLim, cMap, opt, movieType)

    fig = figure('Color', opt.FigureColor, 'Units', 'pixels', ...
        'Position', opt.MovieFigurePosition);
    ax = axes('Parent', fig);

    if opt.ColorMaterialByStrain
        if opt.ShowMovieColorbar
            ax.Position = [0.08 0.08 0.74 0.84];
        else
            ax.Position = [0.08 0.08 0.84 0.84];
        end

        colormap(ax, cMap);
        caxis(ax, cLim);
        apply_color_axis_scale(ax, opt);

        if opt.ShowMovieColorbar
            cb = colorbar(ax, opt.ColorbarLocation);
            cb.TickLabelInterpreter = 'latex';
            cb.Label.Interpreter = 'latex';
            cb.Label.String = sanitize_latex_label(opt.ColorbarLabel);
            cb.Label.Rotation = 270;
            cb.FontSize = opt.ColorbarFontSize;
            cb.Label.FontSize = opt.ColorbarLabelFontSize;
        end
    end

    if isempty(opt.DelayTime)
        delayTime = 1/max(opt.FrameRate, eps);
    else
        delayTime = opt.DelayTime;
    end

    if strcmp(movieType, 'mp4')
        vw = VideoWriter(outputFile, opt.MP4Profile);
        vw.FrameRate = opt.FrameRate;
        if isprop(vw, 'Quality')
            vw.Quality = opt.MP4Quality;
        end
        open(vw);
    else
        tmpGif = [tempname '.gif'];
        wroteGif = false;
    end

    movieFrameSize = [];
    for kk = 1:numel(frameIdx)
        it = frameIdx(kk);

        cla(ax);

        render_one_tile(ax, frameData{it}, ep(:,it), R(it), N, t(it), ...
            Rref, rOuterDisplay, cLim, cMap, opt);

        % Re-enforce color settings after cla/rendering.
        if opt.ColorMaterialByStrain
            colormap(ax, cMap);
            caxis(ax, cLim);
            apply_color_axis_scale(ax, opt);
        end

        drawnow;

        fr = getframe(fig);
        [fr, movieFrameSize] = normalize_movie_frame_size(fr, movieFrameSize);

        if strcmp(movieType, 'mp4')
            writeVideo(vw, fr);
        else
            im = frame2im(fr);
            [A,map] = rgb2ind(im, 256, 'nodither');

            if ~wroteGif
                imwrite(A, map, tmpGif, 'gif', ...
                    'WriteMode', 'overwrite', ...
                    'LoopCount', inf, ...
                    'DelayTime', delayTime);
                wroteGif = true;
            else
                imwrite(A, map, tmpGif, 'gif', ...
                    'WriteMode', 'append', ...
                    'DelayTime', delayTime);
            end
        end

        if opt.Verbose
            fprintf('Rendered frame %d/%d, it = %d, t = %.6g\n', ...
                kk, numel(frameIdx), it, t(it));
        end
    end

    if strcmp(movieType, 'mp4')
        close(vw);
    elseif wroteGif
        if exist(outputFile, 'file') == 2
            delete(outputFile);
        end
        movefile(tmpGif, outputFile, 'f');
    end
end

function [fr, targetSize] = normalize_movie_frame_size(fr, targetSize)
    frameSize = size(fr.cdata);
    frameSize = frameSize(1:2);

    if isempty(targetSize)
        targetSize = frameSize;
        return;
    end

    if ~isequal(frameSize, targetSize)
        fr.cdata = resize_frame_nearest(fr.cdata, targetSize);
    end
end

function imgOut = resize_frame_nearest(imgIn, targetSize)
    srcH = size(imgIn, 1);
    srcW = size(imgIn, 2);
    rowIdx = round(linspace(1, srcH, targetSize(1)));
    colIdx = round(linspace(1, srcW, targetSize(2)));
    imgOut = imgIn(rowIdx, colIdx, :);
end

function render_one_tile(ax, fd, epNow, Rcur, N, tNow, Rref, rOuterDisplay, cLim, cMap, opt)
    hold(ax, 'on');
    if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
        ax.Toolbar.Visible = 'off';
    end
    if strcmpi(char(opt.AxesAspect), 'fill')
        axis(ax, 'normal');
    else
        axis(ax, 'equal');
    end
    axis(ax, [opt.DisplayXLim opt.DisplayZLim]);
    axis(ax, 'manual');
    if opt.ShowAxesBox
        box(ax, 'on');
    else
        box(ax, 'off');
        ax.XColor = 'none';
        ax.YColor = 'none';
    end
    ax.XTick = [];
    ax.YTick = [];

    % Bubble fill and outline.  Compute this before the color field so the
    % color mesh and bubble patch use exactly the same angular discretization.
    phiB = linspace(0, 2*pi, display_phi_count(opt));
    [xBubble, zBubble] = display_boundary_curve(phiB, epNow, Rcur, N, opt);

    % Draw a white bubble below the field first.  The color field is then drawn
    % up to the interface instead of being masked by a white polygon drawn on
    % top of it.  A slightly contracted white fill is drawn after the field to
    % keep the bubble interior clean without covering the exterior wall values.
    if opt.FillBubble
        fill(ax, xBubble, zBubble, opt.BubbleFillColor, 'EdgeColor', 'none');
    end

    if opt.ColorMaterialByStrain
        [X, Z, C] = current_almansi_colour_surface(fd, epNow, Rcur, Rref, N, rOuterDisplay, opt);
        C = prepare_color_data_for_scale(C, cLim, opt);
        if opt.ClampColorDataToCLim && numel(cLim) == 2
            C = min(max(C, cLim(1)), cLim(2));
        end
        if opt.UseContourf
            contourf(ax, X, Z, C, contour_levels_for_scale(cLim, opt), 'LineStyle', 'none');
        else
            surf(ax, X, Z, zeros(size(X)), C, 'EdgeColor', 'none', 'FaceColor', 'interp', 'FaceAlpha', opt.StrainAlpha);
            view(ax, 2);
        end
        colormap(ax, cMap);
        caxis(ax, cLim);
        apply_color_axis_scale(ax, opt);
    end

    % Redraw only the bubble interior, slightly inside the true interface.
    % This prevents the fill polygon from clipping the first colored cells
    % outside a corrugated wall, which was producing jagged colored slivers.
    if opt.FillBubble && opt.ColorMaterialByStrain
        inset = max(0, min(0.05, opt.BubbleFillInsetFrac));
        xInner = (1 - inset) .* xBubble;
        zInner = (1 - inset) .* zBubble;
        fill(ax, xInner, zInner, opt.BubbleFillColor, 'EdgeColor', 'none');
    end

    if opt.FEM_grid
        switch lower(char(opt.GridType))
            case {'material','deformed','lagrangian'}
                draw_remeshed_total_solution_grid(ax, fd, epNow, Rcur, Rref, N, rOuterDisplay, opt, opt.GridCircleSpacing);
            case {'eulerian','current','deformed_current','current_deformed'}
                draw_remeshed_total_solution_grid(ax, fd, epNow, Rcur, Rref, N, rOuterDisplay, opt, 'current');
            case {'pushed','pushed_forward','pushed_forward_raw','raw_material'}
                draw_pushed_forward_material_grid(ax, fd, epNow, Rcur, Rref, N, rOuterDisplay, opt, opt.GridCircleSpacing);
            case {'reference','polar','undeformed','overlay'}
                draw_eulerian_reference_grid(ax, epNow, Rcur, N, rOuterDisplay, opt);
            otherwise
                error('Unknown GridType. Use ''material'', ''eulerian'', ''pushed_forward_raw'', or ''reference''.');
        end
    end

    plot(ax, xBubble, zBubble, '-', 'Color', opt.BubbleColor, 'LineWidth', opt.BubbleWidth);

    if opt.ShowTitle
        if ~isempty(opt.tc) && isfinite(opt.tc) && opt.tc ~= 0
            title(ax, sprintf('$t$ = %.1f', tNow/opt.tc), 'Interpreter', 'latex', 'FontSize', opt.TitleFontSize);
        else
            title(ax, sprintf('$t$ = %.1e', tNow), 'Interpreter', 'latex', 'FontSize', opt.TitleFontSize);
        end
    end
end

function [X, Z, C] = current_almansi_colour_surface(fd, epNow, Rcur, Rref, N, rOuterDisplay, opt)
    % Historical name retained for compatibility. Depending on StrainMeasure,
    % this returns either the original strain colour surface or the Eulerian
    % rate-of-strain colour surface.
    phiPlot = linspace(0, 2*pi, display_phi_count(opt));

    [theta, az, Xfun, Zfun, rb] = make_display_polar_grid(phiPlot, epNow, Rcur, N, opt);
    rb = max(rb, 0.05*Rcur);

    s = linspace(0, 1, opt.StrainMeshR).';
    s = s .^ opt.RadialSpacingPower;

    % Rplot is the radius used to draw the coloured surface.  By default the
    % colour geometry reaches inward dimples of the displayed perturbed
    % interface.  Set MaskBelowMeanWall=true to hide rows that are below the
    % solved mean wall.
    Rplot = rb + (rOuterDisplay - rb) .* s;

    % Rquery is the radius at which the theory is evaluated.  The governing
    % fields T,J,K and the spherical inverse map are defined outside the mean
    % spherical wall r = R(t).  When the displayed perturbed wall has inward
    % dimples, rb < R(t), evaluating the continuum formulas at Rplot < R(t)
    % is outside the solved/linearized domain and can create artificial huge
    % near-wall values.  The default is therefore a constant normal extension
    % of the mean-wall value into the small plotting-only gap rb <= r < R(t),
    % while still drawing that gap so the colour reaches the actual wall.
    Rquery = Rplot;
    switch lower(strrep(char(opt.WallFieldEvaluationMode), '-', '_'))
        case {'mean_wall_extension','meanwall','mean_wall','clamp_to_mean_wall'}
            Rquery = max(Rquery, Rcur);
        case {'perturbed_wall','perturbed','raw','none'}
            % legacy behavior: evaluate directly at the displayed radius
        otherwise
            error('Unknown WallFieldEvaluationMode: %s. Use ''mean_wall_extension'' or ''perturbed_wall''.', char(opt.WallFieldEvaluationMode));
    end

    X = Xfun(Rplot);
    Z = Zfun(Rplot);

    if is_stress_requested(opt)
        C = eulerian_stress_scalar_current(Rquery, theta, az, fd, epNow, Rcur, Rref, N, opt);
    elseif is_rate_of_strain_requested(opt)
        C = eulerian_rate_of_strain_scalar_current(Rquery, theta, az, fd, opt);
    else
        C = eulerian_almansi_scalar_current(Rquery, theta, az, fd, epNow, Rcur, Rref, N, opt);
    end

    if size(C,1) >= 2
        bad = ~isfinite(C(1,:));
        C(1,bad) = C(2,bad);
    end

    if opt.FillMissingColorData
        C = fillmissing_nearest_2d(C);
    end

    if opt.MaskBelowMeanWall
        belowMeanWall = Rplot < Rcur - max(1e-12, 1e-10*max(abs(Rcur),1));
        X(belowMeanWall) = NaN;
        Z(belowMeanWall) = NaN;
        C(belowMeanWall) = NaN;
    end
end

function C = eulerian_almansi_scalar_current(rGrid, thetaVec, azVec, fd, epNow, Rcur, Rref, N, opt)
    % Compute strain from the spatial displacement gradient in the orthonormal
    % spherical basis. Supports axisymmetric m=0 and real sectoral m=n slices.

    [Nr, Nt] = size(rGrid);
    thetaVec = thetaVec(:).';
    azVec = azVec(:).';

    r = real(rGrid);
    rSafe = max(r, eps);

    % Base spherical displacement u_s(r) = r - rho(r).
    rhoCube = rSafe.^3 - Rcur.^3 + Rref.^3;
    rho = nthroot(max(rhoCube, eps), 3);
    ur = rSafe - rho;
    dur_dr = 1 - (rSafe.^2 ./ max(rho.^2, eps));

    uth = zeros(Nr, Nt);
    uph = zeros(Nr, Nt);

    dur_dtheta = zeros(Nr, Nt);
    dur_dphi   = zeros(Nr, Nt);

    duth_dr     = zeros(Nr, Nt);
    duth_dtheta = zeros(Nr, Nt);
    duth_dphi   = zeros(Nr, Nt);

    duph_dr     = zeros(Nr, Nt);
    duph_dtheta = zeros(Nr, Nt);
    duph_dphi   = zeros(Nr, Nt);

    interpMethod = char(opt.RadialInterp);
    s1 = sin(thetaVec);
    c1 = cos(thetaVec);
    s1safe = max(abs(s1), opt.ThetaPoleEps);

    for j = 1:numel(N)
        n = N(j);
        [Y, Yt, Ytt, Yp, Ytp, Ypp] = harmonic_values_for_plot(n, thetaVec, azVec, opt);

        rq = rSafe;

        A  = interp_radial_current(fd.rEval, fd.A(j,:), rq, interpMethod, opt);
        Ar = interp_radial_current(fd.rEval, fd.Ar(j,:), rq, interpMethod, opt);
        B  = interp_radial_current(fd.rEval, fd.B(j,:), rq, interpMethod, opt);
        Br = interp_radial_current(fd.rEval, fd.Br(j,:), rq, interpMethod, opt);

        ur = ur + A .* Y;
        dur_dr = dur_dr + Ar .* Y;
        dur_dtheta = dur_dtheta + A .* Yt;
        dur_dphi   = dur_dphi   + A .* Yp;

        uth = uth + B .* Yt;
        duth_dr = duth_dr + Br .* Yt;
        duth_dtheta = duth_dtheta + B .* Ytt;
        duth_dphi   = duth_dphi   + B .* Ytp;

        uph = uph + B .* (Yp ./ s1safe);
        duph_dr = duph_dr + Br .* (Yp ./ s1safe);
        duph_dtheta = duph_dtheta + B .* (Ytp ./ s1safe - Yp .* c1 ./ (s1safe.^2));
        duph_dphi   = duph_dphi   + B .* (Ypp ./ s1safe);
    end

    [TH, ~] = meshgrid(thetaVec, 1:Nr);
    sTH = sin(TH);
    cTH = cos(TH);
    sTHsafe = max(abs(sTH), opt.ThetaPoleEps);
    cotTH = cTH ./ sTHsafe;

    % Full spatial displacement gradient in orthonormal spherical basis.
    hrr = dur_dr;
    hrth = dur_dtheta ./ rSafe - uth ./ rSafe;
    hrp  = dur_dphi ./ (rSafe .* sTHsafe) - uph ./ rSafe;

    hthr = duth_dr;
    hthth = ur ./ rSafe + duth_dtheta ./ rSafe;
    hthp  = duth_dphi ./ (rSafe .* sTHsafe) - uph .* cotTH ./ rSafe;

    hphr = duph_dr;
    hphth = duph_dtheta ./ rSafe;
    hpp = duph_dphi ./ (rSafe .* sTHsafe) + ur ./ rSafe + uth .* cotTH ./ rSafe;

    switch lower(char(opt.StrainMeasure))
        case {'small','linear','infinitesimal','engineering'}
            err = hrr;
            ett = hthth;
            epp = hpp;
            ert = 0.5 * (hrth + hthr);
            erp = 0.5 * (hrp  + hphr);
            etp = 0.5 * (hthp + hphth);

        case {'almansi','eulerian_almansi','eulerian','finite'}
            % e = 1/2*(I - b^{-1}), b^{-1} = (I-h)^T*(I-h).
            m11 = 1 - hrr;
            m12 = -hrth;
            m13 = -hrp;
            m21 = -hthr;
            m22 = 1 - hthth;
            m23 = -hthp;
            m31 = -hphr;
            m32 = -hphth;
            m33 = 1 - hpp;

            binv11 = m11.^2 + m21.^2 + m31.^2;
            binv12 = m11.*m12 + m21.*m22 + m31.*m32;
            binv13 = m11.*m13 + m21.*m23 + m31.*m33;
            binv22 = m12.^2 + m22.^2 + m32.^2;
            binv23 = m12.*m13 + m22.*m23 + m32.*m33;
            binv33 = m13.^2 + m23.^2 + m33.^2;

            err = 0.5 * (1 - binv11);
            ett = 0.5 * (1 - binv22);
            epp = 0.5 * (1 - binv33);
            ert = -0.5 * binv12;
            erp = -0.5 * binv13;
            etp = -0.5 * binv23;

        otherwise
            error('Unknown StrainMeasure option: %s. Use ''almansi'', ''small'', or ''strain_rate''.', char(opt.StrainMeasure));
    end

    C = strain_scalar_from_components(err, ett, epp, ert, erp, etp, opt.StrainScalar);
    C(~isfinite(C)) = NaN;
end

function C = eulerian_rate_of_strain_scalar_current(rGrid, thetaVec, azVec, fd, opt)
    [drr, dtt, dpp, drt, drp, dtp] = eulerian_rate_of_strain_components_current(rGrid, thetaVec, azVec, fd, opt);
    C = strain_scalar_from_components(drr, dtt, dpp, drt, drp, dtp, active_tensor_scalar(opt));
    C(~isfinite(C)) = NaN;
end

function [drr, dtt, dpp, drt, drp, dtp] = eulerian_rate_of_strain_components_current(rGrid, thetaVec, azVec, fd, opt)
    % Compute the Eulerian rate-of-strain tensor
    %   D = 1/2*(grad(v) + grad(v)^T)
    % in the orthonormal spherical basis, using the same modal angular
    % reconstruction as the strain field.

    [Nr, Nt] = size(rGrid);
    thetaVec = thetaVec(:).';
    azVec = azVec(:).';

    r = real(rGrid);
    rSafe = max(r, eps);

    interpMethod = char(opt.RadialInterp);
    s1 = sin(thetaVec);
    c1 = cos(thetaVec);
    s1safe = max(abs(s1), opt.ThetaPoleEps);

    [TH, ~] = meshgrid(thetaVec, 1:Nr);
    sTH = sin(TH);
    cTH = cos(TH);
    sTHsafe = max(abs(sTH), opt.ThetaPoleEps);
    cotTH = cTH ./ sTHsafe;

    baseFlux = fd.Rcur.^2 .* fd.Rdot;
    vs = baseFlux ./ (rSafe.^2);
    dvs_dr = -2 .* baseFlux ./ (rSafe.^3); %#ok<NASGU>

    vr = vs;
    vth = zeros(Nr, Nt);
    vph = zeros(Nr, Nt);

    dvr_dtheta = zeros(Nr, Nt);
    dvr_dphi   = zeros(Nr, Nt);
    dvth_dtheta = zeros(Nr, Nt);
    dvth_dphi   = zeros(Nr, Nt);
    dvph_dtheta = zeros(Nr, Nt);
    dvph_dphi   = zeros(Nr, Nt);

    for j = 1:numel(fd.N)
        n = fd.N(j);
        [Y, Yt, Ytt, Yp, Ytp, Ypp] = harmonic_values_for_plot(n, thetaVec, azVec, opt);

        rq = rSafe;

        A  = interp_radial_current(fd.rEval, fd.A(j,:), rq, interpMethod, opt);
        Ar = interp_radial_current(fd.rEval, fd.Ar(j,:), rq, interpMethod, opt);
        B  = interp_radial_current(fd.rEval, fd.B(j,:), rq, interpMethod, opt);
        Br = interp_radial_current(fd.rEval, fd.Br(j,:), rq, interpMethod, opt);
        At = interp_radial_current(fd.rEval, fd.At(j,:), rq, interpMethod, opt);
        Bt = interp_radial_current(fd.rEval, fd.Bt(j,:), rq, interpMethod, opt);

        % Velocity coefficients corresponding to the time derivative of the
        % displacement maps plus the base-flow correction in current spherical
        % coordinates.
        Ur = At + vs .* Ar - (-2 .* baseFlux ./ (rSafe.^3)) .* A;
        Ut = Bt + vs .* Br - vs .* B ./ rSafe;

        vr = vr + Ur .* Y;
        dvr_dtheta = dvr_dtheta + Ur .* Yt;
        dvr_dphi   = dvr_dphi   + Ur .* Yp;

        vth = vth + Ut .* Yt;
        dvth_dtheta = dvth_dtheta + Ut .* Ytt;
        dvth_dphi   = dvth_dphi   + Ut .* Ytp;

        vph = vph + Ut .* (Yp ./ s1safe);
        dvph_dtheta = dvph_dtheta + Ut .* (Ytp ./ s1safe - Yp .* c1 ./ (s1safe.^2));
        dvph_dphi   = dvph_dphi   + Ut .* (Ypp ./ s1safe);
    end

    dvr_dr = radial_gradient_columns(vr, rSafe);
    dvth_dr = radial_gradient_columns(vth, rSafe);
    dvph_dr = radial_gradient_columns(vph, rSafe);

    % grad(v) in an orthonormal spherical basis.
    grr = dvr_dr;
    grth = dvr_dtheta ./ rSafe - vth ./ rSafe;
    grp  = dvr_dphi ./ (rSafe .* sTHsafe) - vph ./ rSafe;

    gthr = dvth_dr;
    gthth = vr ./ rSafe + dvth_dtheta ./ rSafe;
    gthp  = dvth_dphi ./ (rSafe .* sTHsafe) - vph .* cotTH ./ rSafe;

    gphr = dvph_dr;
    gphth = dvph_dtheta ./ rSafe;
    gpp = dvph_dphi ./ (rSafe .* sTHsafe) + vr ./ rSafe + vth .* cotTH ./ rSafe;

    drr = grr;
    dtt = gthth;
    dpp = gpp;
    drt = 0.5 * (grth + gthr);
    drp = 0.5 * (grp  + gphr);
    dtp = 0.5 * (gthp + gphth);
end

function C = eulerian_stress_scalar_current(rGrid, thetaVec, azVec, fd, epNow, Rcur, Rref, N, opt)
    % Constitutive stress from the paper:
    %   sigma = G*(1 + alpha*(I_B - 3))*B + 2*mu*D - P*I.
    % If P is omitted, this returns the constitutive material part.  For
    % shear components, this is identical to the full Cauchy stress.

    [err, ett, epp, ert, erp, etp] = eulerian_almansi_components_current(rGrid, thetaVec, azVec, fd, epNow, Rcur, Rref, N, opt);
    [drr, dtt, dpp, drt, drp, dtp] = eulerian_rate_of_strain_components_current(rGrid, thetaVec, azVec, fd, opt);

    % B^{-1} = I - 2e, so B = inv(I - 2e).
    m11 = 1 - 2*err;
    m22 = 1 - 2*ett;
    m33 = 1 - 2*epp;
    m12 = -2*ert;
    m13 = -2*erp;
    m23 = -2*etp;

    detM = m11.*(m22.*m33 - m23.^2) - m12.*(m12.*m33 - m13.*m23) + m13.*(m12.*m23 - m13.*m22);
    smallDet = abs(detM) < 1e-14;
    detM(smallDet) = NaN;

    Brr = (m22.*m33 - m23.^2) ./ detM;
    Btt = (m11.*m33 - m13.^2) ./ detM;
    Bpp = (m11.*m22 - m12.^2) ./ detM;
    Brt = (m13.*m23 - m12.*m33) ./ detM;
    Brp = (m12.*m23 - m13.*m22) ./ detM;
    Btp = (m12.*m13 - m11.*m23) ./ detM;

    IB = Brr + Btt + Bpp;

    Gnow = value_at_frame(opt.G, fd.it, 'G');
    munow = value_at_frame(opt.mu, fd.it, 'mu');
    alphanow = value_at_frame(opt.alpha, fd.it, 'alpha');

    stiffFactor = Gnow .* (1 + alphanow .* (IB - 3));

    srr = stiffFactor .* Brr + 2*munow.*drr;
    stt = stiffFactor .* Btt + 2*munow.*dtt;
    spp = stiffFactor .* Bpp + 2*munow.*dpp;
    srt = stiffFactor .* Brt + 2*munow.*drt;
    srp = stiffFactor .* Brp + 2*munow.*drp;
    stp = stiffFactor .* Btp + 2*munow.*dtp;

    if ~isempty(opt.Pressure)
        Pnow = value_at_frame(opt.Pressure, fd.it, 'Pressure');
        srr = srr - Pnow;
        stt = stt - Pnow;
        spp = spp - Pnow;
    end

    C = strain_scalar_from_components(srr, stt, spp, srt, srp, stp, active_tensor_scalar(opt));
    C(~isfinite(C)) = NaN;
end

function [err, ett, epp, ert, erp, etp] = eulerian_almansi_components_current(rGrid, thetaVec, azVec, fd, epNow, Rcur, Rref, N, opt)
    % Component version of eulerian_almansi_scalar_current.  This intentionally
    % mirrors the original strain path so stress uses the same B/e kinematics.

    [Nr, Nt] = size(rGrid);
    thetaVec = thetaVec(:).';
    azVec = azVec(:).';

    r = real(rGrid);
    rSafe = max(r, eps);

    rhoCube = rSafe.^3 - Rcur.^3 + Rref.^3;
    rho = nthroot(max(rhoCube, eps), 3);
    ur = rSafe - rho;
    dur_dr = 1 - (rSafe.^2 ./ max(rho.^2, eps));

    uth = zeros(Nr, Nt);
    uph = zeros(Nr, Nt);
    dur_dtheta = zeros(Nr, Nt);
    dur_dphi   = zeros(Nr, Nt);
    duth_dr     = zeros(Nr, Nt);
    duth_dtheta = zeros(Nr, Nt);
    duth_dphi   = zeros(Nr, Nt);
    duph_dr     = zeros(Nr, Nt);
    duph_dtheta = zeros(Nr, Nt);
    duph_dphi   = zeros(Nr, Nt);

    interpMethod = char(opt.RadialInterp);
    s1 = sin(thetaVec);
    c1 = cos(thetaVec);
    s1safe = max(abs(s1), opt.ThetaPoleEps);

    for j = 1:numel(N)
        n = N(j);
        [Y, Yt, Ytt, Yp, Ytp, Ypp] = harmonic_values_for_plot(n, thetaVec, azVec, opt);

        rq = rSafe;

        A  = interp_radial_current(fd.rEval, fd.A(j,:), rq, interpMethod, opt);
        Ar = interp_radial_current(fd.rEval, fd.Ar(j,:), rq, interpMethod, opt);
        B  = interp_radial_current(fd.rEval, fd.B(j,:), rq, interpMethod, opt);
        Br = interp_radial_current(fd.rEval, fd.Br(j,:), rq, interpMethod, opt);

        ur = ur + A .* Y;
        dur_dr = dur_dr + Ar .* Y;
        dur_dtheta = dur_dtheta + A .* Yt;
        dur_dphi   = dur_dphi   + A .* Yp;

        uth = uth + B .* Yt;
        duth_dr = duth_dr + Br .* Yt;
        duth_dtheta = duth_dtheta + B .* Ytt;
        duth_dphi   = duth_dphi   + B .* Ytp;

        uph = uph + B .* (Yp ./ s1safe);
        duph_dr = duph_dr + Br .* (Yp ./ s1safe);
        duph_dtheta = duph_dtheta + B .* (Ytp ./ s1safe - Yp .* c1 ./ (s1safe.^2));
        duph_dphi   = duph_dphi   + B .* (Ypp ./ s1safe);
    end

    [TH, ~] = meshgrid(thetaVec, 1:Nr);
    sTH = sin(TH);
    cTH = cos(TH);
    sTHsafe = max(abs(sTH), opt.ThetaPoleEps);
    cotTH = cTH ./ sTHsafe;

    hrr = dur_dr;
    hrth = dur_dtheta ./ rSafe - uth ./ rSafe;
    hrp  = dur_dphi ./ (rSafe .* sTHsafe) - uph ./ rSafe;
    hthr = duth_dr;
    hthth = ur ./ rSafe + duth_dtheta ./ rSafe;
    hthp  = duth_dphi ./ (rSafe .* sTHsafe) - uph .* cotTH ./ rSafe;
    hphr = duph_dr;
    hphth = duph_dtheta ./ rSafe;
    hpp = duph_dphi ./ (rSafe .* sTHsafe) + ur ./ rSafe + uth .* cotTH ./ rSafe;

    sm = lower(char(opt.StrainMeasure));
    if is_stress_requested(opt)
        sm = 'almansi'; % stress elasticity uses B obtained from Eulerian Almansi strain.
    end

    switch sm
        case {'small','linear','infinitesimal','engineering'}
            err = hrr;
            ett = hthth;
            epp = hpp;
            ert = 0.5 * (hrth + hthr);
            erp = 0.5 * (hrp  + hphr);
            etp = 0.5 * (hthp + hphth);

        case {'almansi','eulerian_almansi','eulerian','finite'}
            m11 = 1 - hrr;
            m12 = -hrth;
            m13 = -hrp;
            m21 = -hthr;
            m22 = 1 - hthth;
            m23 = -hthp;
            m31 = -hphr;
            m32 = -hphth;
            m33 = 1 - hpp;

            binv11 = m11.^2 + m21.^2 + m31.^2;
            binv12 = m11.*m12 + m21.*m22 + m31.*m32;
            binv13 = m11.*m13 + m21.*m23 + m31.*m33;
            binv22 = m12.^2 + m22.^2 + m32.^2;
            binv23 = m12.*m13 + m22.*m23 + m32.*m33;
            binv33 = m13.^2 + m23.^2 + m33.^2;

            err = 0.5 * (1 - binv11);
            ett = 0.5 * (1 - binv22);
            epp = 0.5 * (1 - binv33);
            ert = -0.5 * binv12;
            erp = -0.5 * binv13;
            etp = -0.5 * binv23;
        otherwise
            error('Unknown StrainMeasure option: %s. Use ''almansi'', ''small'', ''strain_rate'', or ''stress''.', char(opt.StrainMeasure));
    end
end

function fd = build_frame_modal_data_current(it, T, ep, R, t, N, L, xg, W, w, One_wT, opt)
    xN = size(T,2);
    nModes = numel(N);
    Nt = size(T,3);
    rEval = real(f_r(xg, R(it), L));
    drdx  = real(f_ds(xg, R(it), L));

    A_coeff = zeros(nModes, xN);
    B_coeff = zeros(nModes, xN);
    Ar_coeff = zeros(nModes, xN);
    Br_coeff = zeros(nModes, xN);

    for j = 1:nModes
        n = N(j);
        Tcol = squeeze(T(j,:,it));
        Tcol = real(Tcol(:));

        [KmT, JmT] = f_J_K_kernel(rEval, drdx, One_wT, W, n);
        wkap = f_kappa_kernel(rEval, drdx, n, w);

        K = KmT * Tcol;
        J = JmT * Tcol;
        kappa = wkap * Tcol;

        Phi = ((n+1)/(2*n+1)) .* K .* rEval.^n + ...
              ((n/(n+1)) * R(it)^(2*n+1) * kappa + (n/(2*n+1)) .* J) .* rEval.^(-(n+1));

        ur_rot = n * R(it)^(2*n+1) * kappa .* rEval.^(-(n+2)) + ...
                 (n*(n+1)/(2*n+1)) .* (J .* rEval.^(-(n+2)) - K .* rEval.^(n-1));
        ur_rot(1) = 0;  % T - dPhi/dr vanishes at the mean wall

        Pn = ep(j,it) * R(it)^(n+3);
        A = Pn .* rEval.^(-(n+2)) + ur_rot;
        B = -( Pn ./ ((n+1) .* rEval.^(n+2)) + Phi ./ rEval );

        Arow = real(A(:)).';
        Brow = real(B(:)).';
        A_coeff(j,:) = Arow;
        B_coeff(j,:) = Brow;

        Ar_coeff(j,:) = safe_gradient_1d(Arow, rEval(:).');
        Br_coeff(j,:) = safe_gradient_1d(Brow, rEval(:).');
    end

    % Time derivatives of modal displacement coefficients at fixed current r.
    % For accurate rate-of-strain plots, use the solver quantities Td=dT/dt,
    % epd=dep/dt, and Rd=dR/dt when they are supplied. Otherwise fall back to
    % finite differences in the saved output frames.
    [At_coeff, Bt_coeff, Rdot] = modal_AB_time_derivatives_current( ...
        it, rEval, drdx, T, ep, R, t, N, L, xg, W, w, One_wT, opt);

    fd.rEval = real(rEval(:).');
    fd.A = A_coeff;
    fd.B = B_coeff;
    fd.Ar = Ar_coeff;
    fd.Br = Br_coeff;
    fd.At = At_coeff;
    fd.Bt = Bt_coeff;
    fd.N = N(:).';
    fd.it = it;
    fd.Rcur = R(it);
    fd.Rdot = Rdot;
end

function [At_coeff, Bt_coeff, Rdot] = modal_AB_time_derivatives_current( ...
    it, rEval, drdx, T, ep, R, t, N, L, xg, W, w, One_wT, opt)
    nModes = numel(N);
    xN = size(T,2);
    rRow = real(rEval(:).');

    Rdot = derivative_scalar_at_frame(R, t, it, opt.Rd);

    if ~isempty(opt.Td)
        TdNow = squeeze(opt.Td(:,:,it));
        if nModes == 1
            TdNow = reshape(TdNow, [1, xN]);
        end
        epdNow = derivative_rows_at_frame(ep, t, it, opt.epd);
        [At_coeff, Bt_coeff] = modal_AB_time_derivatives_from_rates( ...
            it, rEval, drdx, T, TdNow, ep, epdNow, R, Rdot, N, W, w, One_wT);
        return;
    end

    if ~opt.UseFiniteDifferenceForRate
        error(['Rate-of-strain plotting requires Td/Tdot/V unless ', ...
               '''UseFiniteDifferenceForRate'' is true.']);
    end

    [At_coeff, Bt_coeff, Rdot] = modal_AB_time_derivatives_finite_difference( ...
        it, rRow, T, ep, R, t, N, L, xg, W, w, One_wT);
end

function [At_coeff, Bt_coeff] = modal_AB_time_derivatives_from_rates( ...
    it, rEval, drdx, T, TdNow, ep, epdNow, R, Rdot, N, W, w, One_wT)
    nModes = numel(N);
    xN = size(T,2);
    rCol = real(rEval(:));

    At_coeff = zeros(nModes, xN);
    Bt_coeff = zeros(nModes, xN);

    for j = 1:nModes
        n = N(j);
        Tcol  = real(squeeze(T(j,:,it))); Tcol = Tcol(:);
        Tdcol = real(squeeze(TdNow(j,:))); Tdcol = Tdcol(:);

        [KmT, JmT] = f_J_K_kernel(rEval, drdx, One_wT, W, n);
        wkap = f_kappa_kernel(rEval, drdx, n, w);

        K = KmT * Tcol;
        J = JmT * Tcol;
        kappa = wkap * Tcol;

        Kt = KmT * Tdcol;
        Jt = JmT * Tdcol;
        kappat = wkap * Tdcol;

        % Leibniz corrections from the moving lower limit R(t) in J and kappa.
        % K has lower limit r for fixed-current-r differentiation and therefore
        % does not receive the Rdot*T(R,t) correction.
        Twall = Tcol(1);
        Jt = Jt - Rdot .* R(it).^(n+1) .* Twall;
        kappat = kappat + ((n+1)/(2*n+1)) .* Rdot .* R(it).^(-n) .* Twall;

        commonRate = (2*n + 1) .* R(it).^(2*n) .* Rdot .* kappa + ...
                     R(it).^(2*n+1) .* kappat;

        Phit = ((n+1)/(2*n+1)) .* Kt .* rCol.^n + ...
               ((n/(n+1)) .* commonRate + (n/(2*n+1)) .* Jt) .* rCol.^(-(n+1));

        urrott = n .* commonRate .* rCol.^(-(n+2)) + ...
                 (n*(n+1)/(2*n+1)) .* (Jt .* rCol.^(-(n+2)) - Kt .* rCol.^(n-1));
        urrott(1) = 0;  % wall-normal rotational velocity contribution is zero

        Pnt = (epdNow(j) .* R(it).^(n+3) + ...
               ep(j,it) .* (n+3) .* R(it).^(n+2) .* Rdot);

        At = Pnt .* rCol.^(-(n+2)) + urrott;
        Bt = -( Pnt ./ ((n+1) .* rCol.^(n+2)) + Phit ./ rCol );

        At_coeff(j,:) = real(At(:)).';
        Bt_coeff(j,:) = real(Bt(:)).';
    end
end

function [At_coeff, Bt_coeff, Rdot] = modal_AB_time_derivatives_finite_difference( ...
    it, rQuery, T, ep, R, t, N, L, xg, W, w, One_wT)
    nModes = numel(N);
    xN = size(T,2);
    Nt = size(T,3);

    if Nt == 1
        At_coeff = zeros(nModes, xN);
        Bt_coeff = zeros(nModes, xN);
        Rdot = 0;
        return;
    end

    if it == 1
        im = 1; ip = 2;
    elseif it == Nt
        im = Nt-1; ip = Nt;
    else
        im = it-1; ip = it+1;
    end
    dt = t(ip) - t(im);
    if ~isfinite(dt) || abs(dt) < eps
        At_coeff = zeros(nModes, xN);
        Bt_coeff = zeros(nModes, xN);
        Rdot = 0;
        return;
    end

    [Aminus, Bminus] = modal_AB_on_query_r(im, rQuery, T, ep, R, N, L, xg, W, w, One_wT);
    [Aplus,  Bplus]  = modal_AB_on_query_r(ip, rQuery, T, ep, R, N, L, xg, W, w, One_wT);
    At_coeff = (Aplus - Aminus) ./ dt;
    Bt_coeff = (Bplus - Bminus) ./ dt;
    Rdot = (R(ip) - R(im)) ./ dt;
end

function valdot = derivative_scalar_at_frame(y, t, it, ydotProvided)
    y = y(:);
    if ~isempty(ydotProvided)
        yp = ydotProvided(:);
        valdot = yp(it);
        return;
    end
    Nt = numel(y);
    if Nt == 1
        valdot = 0;
        return;
    end
    if it == 1
        im = 1; ip = 2;
    elseif it == Nt
        im = Nt-1; ip = Nt;
    else
        im = it-1; ip = it+1;
    end
    dt = t(ip) - t(im);
    if ~isfinite(dt) || abs(dt) < eps
        valdot = 0;
    else
        valdot = (y(ip) - y(im)) ./ dt;
    end
end

function epdNow = derivative_rows_at_frame(ep, t, it, epdProvided)
    if ~isempty(epdProvided)
        epdNow = real(epdProvided(:,it));
        return;
    end
    Nt = size(ep,2);
    if Nt == 1
        epdNow = zeros(size(ep,1),1);
        return;
    end
    if it == 1
        im = 1; ip = 2;
    elseif it == Nt
        im = Nt-1; ip = Nt;
    else
        im = it-1; ip = it+1;
    end
    dt = t(ip) - t(im);
    if ~isfinite(dt) || abs(dt) < eps
        epdNow = zeros(size(ep,1),1);
    else
        epdNow = (ep(:,ip) - ep(:,im)) ./ dt;
    end
end

function [Aq, Bq] = modal_AB_on_query_r(it, rQuery, T, ep, R, N, L, xg, W, w, One_wT)
    nModes = numel(N);
    xN = size(T,2);
    rEval = real(f_r(xg, R(it), L));
    drdx  = real(f_ds(xg, R(it), L));
    rEvalRow = rEval(:).';
    rQuery = real(rQuery(:).');

    Aq = zeros(nModes, numel(rQuery));
    Bq = zeros(nModes, numel(rQuery));

    for j = 1:nModes
        n = N(j);
        Tcol = squeeze(T(j,:,it));
        Tcol = real(Tcol(:));

        [KmT, JmT] = f_J_K_kernel(rEval, drdx, One_wT, W, n);
        wkap = f_kappa_kernel(rEval, drdx, n, w);

        K = KmT * Tcol;
        J = JmT * Tcol;
        kappa = wkap * Tcol;

        Phi = ((n+1)/(2*n+1)) .* K .* rEval.^n + ...
              ((n/(n+1)) * R(it)^(2*n+1) * kappa + (n/(2*n+1)) .* J) .* rEval.^(-(n+1));

        ur_rot = n * R(it)^(2*n+1) * kappa .* rEval.^(-(n+2)) + ...
                 (n*(n+1)/(2*n+1)) .* (J .* rEval.^(-(n+2)) - K .* rEval.^(n-1));
        ur_rot(1) = 0;  % T - dPhi/dr vanishes at the mean wall

        Pn = ep(j,it) * R(it)^(n+3);
        A = Pn .* rEval.^(-(n+2)) + ur_rot;
        B = -( Pn ./ ((n+1) .* rEval.^(n+2)) + Phi ./ rEval );

        rq = min(max(rQuery, rEvalRow(1)), rEvalRow(end));
        Aq(j,:) = interp1(rEvalRow, real(A(:)).', rq, 'pchip', 'extrap');
        Bq(j,:) = interp1(rEvalRow, real(B(:)).', rq, 'pchip', 'extrap');
    end

    %#ok<NASGU> xN retained for consistency with the caller's grid size.
end

function G = radial_gradient_columns(F, rGrid)
    G = zeros(size(F));
    for k = 1:size(F,2)
        rr = real(rGrid(:,k));
        ff = real(F(:,k));
        if numel(rr) < 2 || all(abs(diff(rr)) < eps)
            G(:,k) = 0;
        else
            G(:,k) = gradient(ff, rr);
        end
    end
end

function g = safe_gradient_1d(y, x)
    y = y(:).';
    x = x(:).';
    if numel(y) < 2
        g = zeros(size(y));
        return;
    end
    g = gradient(y, x);
    g(~isfinite(g)) = 0;
end


function draw_remeshed_total_solution_grid(ax, fd, epNow, Rcur, Rref, N, rOuterDisplay, opt, radialSpacingMode)
    % First evaluate the complete potential-plus-rotational displacement map.
    % Reparameterizing its radial paths prevents a finite displacement from
    % evacuating every grid level next to the displayed bubble boundary.
    radialSpacingMode = normalize_grid_radial_spacing(radialSpacingMode);
    roWall = Rref;
    roOuter = nthroot(max(rOuterDisplay.^3 - Rcur.^3 + Rref.^3, Rref.^3), 3);

    if isempty(opt.RoGridVals)
        nCircles = max(round(opt.GridCircles), 1);
        if opt.HideWallCircle
            s0 = opt.WallCircleOffsetFrac;
        else
            s0 = 0;
        end
        sRequested = linspace(s0, 1, nCircles) .^ opt.RadialSpacingPower;
        roCircles = material_radii_for_grid_spacing(sRequested, radialSpacingMode, ...
            Rcur, Rref, rOuterDisplay, roWall, roOuter);
    else
        roCircles = real(opt.RoGridVals(:).');
        roCircles = roCircles(isfinite(roCircles) & roCircles >= roWall & roCircles <= roOuter);
        roCircles = unique(roCircles, 'stable');
        if ~opt.HideWallCircle && (isempty(roCircles) || abs(roCircles(1)-roWall) > 1e-12)
            roCircles = [roWall roCircles];
        end
    end

    rCircleBase = nthroot(max(roCircles.^3 + Rcur.^3 - Rref.^3, Rcur.^3), 3);
    sCircles = (rCircleBase - Rcur) ./ max(rOuterDisplay - Rcur, eps);
    sCircles = min(max(real(sCircles), 0), 1);
    if ~opt.HideWallCircle && ~isempty(sCircles)
        sCircles(1) = 0;
    end

    if isempty(opt.ThetaGridEq)
        phiRays = linspace(0, 2*pi, max(round(opt.GridRays), 1)+1);
        phiRays(end) = [];
    else
        phiRays = mod(real(opt.ThetaGridEq(:).'), 2*pi);
        phiRays = unique(phiRays(isfinite(phiRays)), 'stable');
    end

    phiCirc = linspace(0, 2*pi, max(round(opt.PtsPerCircle), 3));
    phiAll = [phiCirc phiRays];
    circleCols = 1:numel(phiCirc);
    rayCols = numel(phiCirc) + (1:numel(phiRays));

    if isempty(opt.GridPathSamples)
        nPath = max([128, 4*max(numel(sCircles), 1), ceil(opt.PtsPerRay/2)]);
    else
        validateattributes(opt.GridPathSamples, {'numeric'}, {'scalar','real','finite','positive'});
        nPath = max(round(opt.GridPathSamples), 32);
    end
    sPath = unique([linspace(0, 1, nPath), sCircles]);
    rBase = Rcur + (rOuterDisplay - Rcur) .* sPath;
    roPath = nthroot(max(rBase.^3 - Rcur.^3 + Rref.^3, Rref.^3), 3);

    if strcmpi(char(opt.AngleChoice), 'reference')
        [Xraw, Zraw] = forward_material_mesh_separable(roPath, phiAll, ...
            fd, Rcur, Rref, N, opt);
    else
        [roMesh, phiMesh] = ndgrid(roPath, phiAll);
        thetaMesh = meridional_theta_from_plot_angle(phiMesh, opt.ThetaPoleEps);
        sxMesh = sign_nonzero(cos(phiMesh));
        rawOpt = opt;
        rawOpt.ClipInsideBubble = false;
        [Xraw, Zraw] = forward_material_line(roMesh, thetaMesh, sxMesh, true(size(roMesh)), ...
            fd, epNow, Rcur, Rref, N, rawOpt, 'mesh', false);
    end
    [Xgrid, Zgrid] = remesh_total_solution_paths(Xraw, Zraw, sPath, phiAll, ...
        epNow, Rcur, N, rOuterDisplay, opt);

    circleX = nan(numel(sCircles), numel(phiCirc)+1);
    circleZ = nan(numel(sCircles), numel(phiCirc)+1);
    for k = 1:numel(sCircles)
        [~, row] = min(abs(sPath - sCircles(k)));
        if sCircles(k) <= 10*eps
            thCirc = meridional_theta_from_plot_angle(phiCirc, opt.ThetaPoleEps);
            sxCirc = sign_nonzero(cos(phiCirc));
            [Xline, Zline] = bubble_curve_level_set(thCirc, sxCirc, epNow, Rcur, N, opt);
        else
            Xline = Xgrid(row, circleCols);
            Zline = Zgrid(row, circleCols);
        end
        circleX(k,1:end-1) = Xline;
        circleZ(k,1:end-1) = Zline;
    end
    if ~isempty(sCircles)
        plot(ax, reshape(circleX.', [], 1), reshape(circleZ.', [], 1), '-', ...
            'Color', opt.GridColor, 'LineWidth', opt.LineWidth);
    end
    if ~isempty(rayCols)
        rayX = [Xgrid(:,rayCols); nan(1,numel(rayCols))];
        rayZ = [Zgrid(:,rayCols); nan(1,numel(rayCols))];
        plot(ax, rayX(:), rayZ(:), '-', 'Color', opt.GridColor, 'LineWidth', opt.LineWidth);
    end
end

function [Xraw, Zraw] = forward_material_mesh_separable(roPath, phiAll, fd, Rcur, Rref, N, opt)
    % The reference-angle map separates exactly into radial coefficients and
    % angular harmonics. Evaluate each unique coordinate once, then assemble
    % the same total first-order displacement with implicit outer products.
    ro = real(roPath(:));
    radicand = ro.^3 + Rcur.^3 - Rref.^3;
    rs = nthroot(max(radicand, 0), 3);
    wallTol = max(1e-9, 1e-7*Rref);
    rs(abs(ro-Rref) <= wallTol) = Rcur;
    rq = min(max(rs, fd.rEval(1)), fd.rEval(end));

    phi = real(phiAll(:).');
    theta = meridional_theta_from_plot_angle(phi, opt.ThetaPoleEps);
    sx = sign_nonzero(cos(phi));
    az = slice_azimuth_from_side(sx, opt);

    urSum = zeros(numel(rs), numel(phi));
    dthSum = zeros(numel(rs), numel(phi));
    for j = 1:numel(N)
        [Y, dY, ~, ~, ~, ~] = harmonic_values_for_plot(N(j), theta, az, opt);
        urCoeff = interp_radial_current(fd.rEval, fd.A(j,:), rq, 'linear', opt);
        uthCoeff = interp_radial_current(fd.rEval, fd.B(j,:), rq, 'linear', opt);
        dthCoeff = uthCoeff(:) ./ max(abs(rs), eps);
        urSum = urSum + urCoeff(:) .* reshape(Y, 1, []);
        dthSum = dthSum + dthCoeff .* reshape(dY, 1, []);
    end

    sinTh = sin(theta);
    cosTh = cos(theta);
    Xraw = sx .* ((rs + urSum) .* sinTh + rs .* dthSum .* cosTh);
    Zraw =       (rs + urSum) .* cosTh - rs .* dthSum .* sinTh;
    Xraw(:,abs(sinTh) < 1e-12) = 0;
end

function [Xgrid, Zgrid] = remesh_total_solution_paths(Xraw, Zraw, sPath, phiAll, epNow, Rcur, N, rOuterDisplay, opt)
    Xraw = real(Xraw);
    Zraw = real(Zraw);
    wallClearance = max(0, opt.GridCircleWallClearanceFrac) * max(abs(Rcur), eps);

    invalidCols = find(~all(isfinite(Xraw) & isfinite(Zraw), 1));
    for j = invalidCols
        finite = isfinite(Xraw(:,j)) & isfinite(Zraw(:,j));
        if nnz(finite) >= 2
            first = find(finite, 1, 'first');
            last = find(finite, 1, 'last');
            validRows = find(finite(first:last)) + first - 1;
            source = linspace(0, 1, numel(validRows));
            Xraw(:,j) = interp1(source, Xraw(validRows,j), sPath(:), 'linear', 'extrap');
            Zraw(:,j) = interp1(source, Zraw(validRows,j), sPath(:), 'linear', 'extrap');
        else
            phi0 = phiAll(j);
            th0 = meridional_theta_from_plot_angle(phi0, opt.ThetaPoleEps);
            sx0 = sign_nonzero(cos(phi0));
            rb0 = bubble_radius_general(th0, sx0, epNow, Rcur, N, opt);
            rr = rb0 + (rOuterDisplay-rb0) .* sPath(:);
            Xraw(:,j) = sx0 .* rr .* sin(th0);
            Zraw(:,j) = rr .* cos(th0);
        end
    end

    thRaw = atan2(abs(Xraw), Zraw);
    sxRaw = sign_nonzero(Xraw);
    rbRaw = bubble_radius_general(thRaw, sxRaw, epNow, Rcur, N, opt);
    rrRaw = hypot(Xraw, Zraw);
    minRawRadius = rbRaw + wallClearance;
    inside = rrRaw < minRawRadius;
    Xraw(inside) = sxRaw(inside) .* minRawRadius(inside) .* sin(thRaw(inside));
    Zraw(inside) = minRawRadius(inside) .* cos(thRaw(inside));

    thFirst = atan2(abs(Xraw(1,:)), Zraw(1,:));
    sxFirst = sign_nonzero(Xraw(1,:));
    rbFirst = bubble_radius_general(thFirst, sxFirst, epNow, Rcur, N, opt);
    Xraw(1,:) = sxFirst .* rbFirst .* sin(thFirst);
    Zraw(1,:) = rbFirst .* cos(thFirst);

    thLast = atan2(abs(Xraw(end,:)), Zraw(end,:));
    sxLast = sign_nonzero(Xraw(end,:));
    Xraw(end,:) = sxLast .* rOuterDisplay .* sin(thLast);
    Zraw(end,:) = rOuterDisplay .* cos(thLast);

    arc = [zeros(1, size(Xraw,2)); cumsum(hypot(diff(Xraw,1,1), diff(Zraw,1,1)), 1)];
    Xgrid = nan(size(Xraw));
    Zgrid = nan(size(Zraw));
    for j = 1:size(Xraw,2)
        [arcCol, ia] = unique(arc(:,j), 'stable');
        x = Xraw(ia,j);
        z = Zraw(ia,j);
        if numel(arcCol) < 2 || arcCol(end) <= eps(max(abs([x; z])))
            Xgrid(:,j) = x(1) + (x(end)-x(1)) .* sPath(:);
            Zgrid(:,j) = z(1) + (z(end)-z(1)) .* sPath(:);
        else
            arcCol = arcCol ./ arcCol(end);
            Xgrid(:,j) = interp1(arcCol, x, sPath(:), 'linear');
            Zgrid(:,j) = interp1(arcCol, z, sPath(:), 'linear');
        end
    end

    thGrid = atan2(abs(Xgrid), Zgrid);
    sxGrid = sign_nonzero(Xgrid);
    rbGrid = bubble_radius_general(thGrid, sxGrid, epNow, Rcur, N, opt);
    rrGrid = hypot(Xgrid, Zgrid);
    clearance = wallClearance .* (sPath(:) > 0);
    minGridRadius = rbGrid + clearance;
    tooClose = rrGrid < minGridRadius;
    Xgrid(tooClose) = sxGrid(tooClose) .* minGridRadius(tooClose) .* sin(thGrid(tooClose));
    Zgrid(tooClose) = minGridRadius(tooClose) .* cos(thGrid(tooClose));
end


function draw_pushed_forward_material_grid(ax, fd, epNow, Rcur, Rref, N, rOuterDisplay, opt, radialSpacingMode)
    % Draw material-labelled circles/rays pushed forward by the same first-order
    % displacement map used in the theory. This is a deformation-grid overlay only;
    % the strain colour field is computed separately on the Eulerian grid.
    if nargin < 9 || isempty(radialSpacingMode)
        radialSpacingMode = opt.GridCircleSpacing;
    end
    radialSpacingMode = normalize_grid_radial_spacing(radialSpacingMode);

    roOuter = nthroot(max(rOuterDisplay.^3 - Rcur.^3 + Rref.^3, Rref.^3), 3);
    roWall = Rref;

    if isempty(opt.RoGridVals)
        nCircles = max(round(opt.GridCircles), 1);
        if opt.HideWallCircle
            s0 = opt.WallCircleOffsetFrac;
        else
            s0 = 0;
        end
        sC = linspace(s0, 1, nCircles);
        sC = sC .^ opt.RadialSpacingPower;
        roVals = material_radii_for_grid_spacing(sC, radialSpacingMode, Rcur, Rref, rOuterDisplay, roWall, roOuter);
        if ~opt.HideWallCircle
            roVals(1) = roWall;
        end
    else
        roVals = real(opt.RoGridVals(:).');
        roVals = roVals(isfinite(roVals) & roVals >= roWall & roVals <= roOuter);
        roVals = unique(roVals, 'stable');
        if ~opt.HideWallCircle && (isempty(roVals) || abs(roVals(1)-roWall) > 1e-12)
            roVals = [roWall roVals];
        end
    end

    if isempty(opt.ThetaGridEq)
        phiRays = linspace(0, 2*pi, opt.GridRays+1);
        phiRays(end) = [];
    else
        phiRays = mod(real(opt.ThetaGridEq(:).'), 2*pi);
        phiRays = unique(phiRays, 'stable');
    end

    phiCirc = linspace(0, 2*pi, opt.PtsPerCircle);
    thCirc = meridional_theta_from_plot_angle(phiCirc, opt.ThetaPoleEps);
    sxCirc = sign_nonzero(cos(phiCirc));

    for k = 1:numel(roVals)
        roLine = roVals(k) * ones(size(phiCirc));
        if abs(roVals(k)-roWall) <= max(1e-12, 1e-10*roWall)
            [Xdef, Zdef] = bubble_curve_level_set(thCirc, sxCirc, epNow, Rcur, N, opt);
        else
            [Xdef, Zdef] = forward_material_line(roLine, thCirc, sxCirc, true(size(roLine)), ...
                fd, epNow, Rcur, Rref, N, opt, 'circle', false);
        end
        plot(ax, Xdef, Zdef, '-', 'Color', opt.GridColor, 'LineWidth', opt.LineWidth);
    end

    sRay = linspace(0, 1, opt.PtsPerRay) .^ opt.RadialSpacingPower;
    roRayBase = material_radii_for_grid_spacing(sRay, radialSpacingMode, Rcur, Rref, rOuterDisplay, roWall, roOuter);
    for k = 1:numel(phiRays)
        phi0 = phiRays(k);
        thLine = meridional_theta_from_plot_angle(phi0, opt.ThetaPoleEps) * ones(size(roRayBase));
        sxLine = sign_nonzero(cos(phi0)) * ones(size(roRayBase));
        [Xdef, Zdef] = forward_material_line(roRayBase, thLine, sxLine, true(size(roRayBase)), ...
            fd, epNow, Rcur, Rref, N, opt, 'ray', opt.AnchorRaysToBubble);
        plot(ax, Xdef, Zdef, '-', 'Color', opt.GridColor, 'LineWidth', opt.LineWidth);
    end
end

function mode = normalize_grid_radial_spacing(modeIn)
    mode = lower(char(modeIn));
    switch mode
        case {'current','display','eulerian'}
            mode = 'current';
        case {'material','reference','lagrangian'}
            mode = 'material';
        otherwise
            error('GridCircleSpacing must be ''current'' or ''material''.');
    end
end

function roVals = material_radii_for_grid_spacing(sVals, mode, Rcur, Rref, rOuterDisplay, roWall, roOuter)
    switch mode
        case 'current'
            rVals = Rcur + (rOuterDisplay - Rcur) .* sVals;
            roVals = nthroot(max(rVals.^3 - Rcur.^3 + Rref.^3, Rref.^3), 3);
        case 'material'
            roVals = roWall + (roOuter - roWall) .* sVals;
        otherwise
            error('Internal error: unknown grid radial spacing mode.');
    end
end

function [Xdef, Zdef] = forward_material_line(ro, thetao, sx, keep, fd, epNow, Rcur, Rref, N, opt, lineKind, anchorRay)
    Xdef = nan(size(ro));
    Zdef = nan(size(ro));

    keep2 = keep & isfinite(ro) & isfinite(thetao) & isfinite(sx);
    if ~any(keep2)
        return;
    end

    ro_o = real(ro(keep2));
    th_o = real(thetao(keep2));
    sx_o = real(sx(keep2));

    radicand = ro_o.^3 + Rcur.^3 - Rref.^3;
    valid = isfinite(radicand) & (radicand >= 0);
    if ~any(valid)
        return;
    end

    ro_o = ro_o(valid);
    th_o = th_o(valid);
    sx_o = sx_o(valid);
    rs = nthroot(radicand(valid), 3);

    wallTol = max(1e-9, 1e-7*Rref);
    onWall = abs(ro_o - Rref) <= wallTol;
    rs(onWall) = Rcur;

    rs_plot = rs;
    rq = min(max(rs, fd.rEval(1)), fd.rEval(end));

    angleChoice = lower(char(opt.AngleChoice));
    theta_use = th_o;
    if strcmp(angleChoice, 'current')
        for iter = 1:max(1, round(opt.AngleIterations))
            [~, dth_tmp] = modal_sums_material(theta_use, sx_o, rs, rq, fd, N, opt);
            theta_use = th_o + dth_tmp;
            theta_use = max(opt.ThetaPoleEps, min(pi-opt.ThetaPoleEps, real(theta_use)));
        end
    end

    if strcmp(angleChoice, 'reference')
        [ur_sum, dth_sum] = modal_sums_material(th_o, sx_o, rs, rq, fd, N, opt);
    else
        [ur_sum, dth_sum] = modal_sums_material(theta_use, sx_o, rs, rq, fd, N, opt);
    end

    % Appendix A gives a first-order coordinate map. Expand the current
    % Cartesian position to the same order so that products such as
    % ur*dtheta and dtheta^2 are not introduced by the renderer.
    sinTh = sin(th_o);
    cosTh = cos(th_o);
    Xtmp = sx_o .* ((rs_plot + ur_sum) .* sinTh + rs_plot .* dth_sum .* cosTh);
    Ztmp =          (rs_plot + ur_sum) .* cosTh - rs_plot .* dth_sum .* sinTh;

    pole = abs(sinTh) < 1e-12;
    Xtmp(pole) = 0;

    if opt.ClipInsideBubble
        rtmp = hypot(Xtmp, Ztmp);
        thtmp = atan2(abs(Xtmp), Ztmp);
        sxTmp = sign_nonzero(Xtmp);
        rb = bubble_radius_general(thtmp, sxTmp, epNow, Rcur, N, opt);
        keepOutside = (strcmp(lineKind, 'circle') && opt.KeepGridCirclesOutsideBubble) || ...
                      (strcmp(lineKind, 'ray') && opt.KeepGridRaysOutsideBubble);
        if keepOutside
            wallClearance = max(0, opt.GridCircleWallClearanceFrac) * max(abs(Rcur), eps);
            lineOffset = max(rs_plot - Rcur, wallClearance);
            minLineRadius = rb + lineOffset;
            tooClose = rtmp < minLineRadius;
            Xtmp(tooClose) = sxTmp(tooClose) .* minLineRadius(tooClose) .* sin(thtmp(tooClose));
            Ztmp(tooClose) = minLineRadius(tooClose) .* cos(thtmp(tooClose));
        else
            inside = rtmp < rb - max(1e-10, 1e-8*max(Rcur,1));
            Xtmp(inside) = NaN;
            Ztmp(inside) = NaN;
        end
    end

    if anchorRay && strcmp(lineKind, 'ray')
        idx0 = find(isfinite(Xtmp) & isfinite(Ztmp), 1, 'first');
        if ~isempty(idx0)
            thAnchor = atan2(abs(Xtmp(idx0)), Ztmp(idx0));
            sxAnchor = sign_nonzero(Xtmp(idx0));
            rbAnchor = bubble_radius_general(thAnchor, sxAnchor, epNow, Rcur, N, opt);
            xb = sxAnchor .* rbAnchor .* sin(thAnchor);
            zb =            rbAnchor .* cos(thAnchor);
            Xtmp(idx0) = xb;
            Ztmp(idx0) = zb;
        end
    end

    idx = find(keep2);
    idx = idx(valid);
    Xdef(idx) = real(Xtmp);
    Zdef(idx) = real(Ztmp);
end

function [ur_sum, dth_sum] = modal_sums_material(theta_eval, sx_eval, rs, rq, fd, N, opt)
    ur_sum = zeros(size(rs));
    dth_sum = zeros(size(rs));
    az = slice_azimuth_from_side(sx_eval, opt);

    for j = 1:numel(N)
        n = N(j);
        [Y, dY, ~, ~, ~, ~] = harmonic_values_for_plot(n, theta_eval, az, opt);
        Y = reshape(Y, size(theta_eval));
        dY = reshape(dY, size(theta_eval));

        % Use the total displacement coefficients shared with the strain path.
        ur_coeff = interp_radial_current(fd.rEval, fd.A(j,:), rq, 'linear', opt);
        uth_coeff = interp_radial_current(fd.rEval, fd.B(j,:), rq, 'linear', opt);
        ur_coeff = reshape(ur_coeff, size(rs));
        uth_coeff = reshape(uth_coeff, size(rs));
        dth_coeff = uth_coeff ./ max(abs(rs), eps);

        ur_sum  = ur_sum  + ur_coeff  .* Y;
        dth_sum = dth_sum + dth_coeff .* dY;
    end
end

function draw_eulerian_reference_grid(ax, epNow, Rcur, N, rOuterDisplay, opt)
    phi = linspace(0, 2*pi, opt.PtsPerCircle);
    [~, ~, Xfun, Zfun, rbCirc] = make_display_polar_grid(phi, epNow, Rcur, N, opt);

    if opt.HideWallCircle
        s0 = opt.WallCircleOffsetFrac;
    else
        s0 = 0;
    end
    sCircles = linspace(s0, 1, max(opt.GridCircles,1));
    sCircles = sCircles .^ opt.RadialSpacingPower;
    for k = 1:numel(sCircles)
        rr = rbCirc + (rOuterDisplay - rbCirc) .* sCircles(k);
        plot(ax, Xfun(rr), Zfun(rr), '-', 'Color', opt.GridColor, 'LineWidth', opt.LineWidth);
    end

    phiRays = linspace(0, 2*pi, opt.GridRays+1);
    phiRays(end) = [];
    sRay = linspace(0,1,opt.PtsPerRay) .^ opt.RadialSpacingPower;
    for k = 1:numel(phiRays)
        phi0 = phiRays(k) * ones(size(sRay));
        [~, ~, XfunR, ZfunR, rb0] = make_display_polar_grid(phi0, epNow, Rcur, N, opt);
        rr = rb0 + (rOuterDisplay - rb0) .* sRay;
        plot(ax, XfunR(rr), ZfunR(rr), '-', 'Color', opt.GridColor, 'LineWidth', opt.LineWidth);
    end
end

function C = strain_scalar_from_components(err, ett, epp, ert, erp, etp, scalarChoice)
    switch lower(char(scalarChoice))
        case {'signed_maxabs_principal','principal_signed','signed_principal'}
            if any(abs(erp(:)) > 0) || any(abs(etp(:)) > 0)
                C = principal_strain_3d(err, ett, epp, ert, erp, etp, 'signed_maxabs');
            else
                tr2 = 0.5 * (err + ett);
                rad2 = sqrt(max(((err - ett) * 0.5).^2 + ert.^2, 0));
                lam1 = tr2 + rad2;
                lam2 = tr2 - rad2;
                lam3 = epp;
                C = lam1;
                pick2 = abs(lam2) > abs(C);
                C(pick2) = lam2(pick2);
                pick3 = abs(lam3) > abs(C);
                C(pick3) = lam3(pick3);
            end

        case {'max_principal','largest_principal'}
            if any(abs(erp(:)) > 0) || any(abs(etp(:)) > 0)
                C = principal_strain_3d(err, ett, epp, ert, erp, etp, 'max');
            else
                tr2 = 0.5 * (err + ett);
                rad2 = sqrt(max(((err - ett) * 0.5).^2 + ert.^2, 0));
                lam1 = tr2 + rad2;
                lam2 = tr2 - rad2;
                lam3 = epp;
                C = max(max(lam1, lam2), lam3);
            end

        case {'min_principal','smallest_principal'}
            if any(abs(erp(:)) > 0) || any(abs(etp(:)) > 0)
                C = principal_strain_3d(err, ett, epp, ert, erp, etp, 'min');
            else
                tr2 = 0.5 * (err + ett);
                rad2 = sqrt(max(((err - ett) * 0.5).^2 + ert.^2, 0));
                lam1 = tr2 + rad2;
                lam2 = tr2 - rad2;
                lam3 = epp;
                C = min(min(lam1, lam2), lam3);
            end

        case {'maxabs_component','max_abs_component','component_max'}
            C = max(max(abs(err), abs(ett)), ...
                max(abs(epp), max(abs(ert), max(abs(erp), abs(etp)))));

        case {'frobenius','norm','magnitude'}
            C = sqrt(err.^2 + ett.^2 + epp.^2 + 2*(ert.^2 + erp.^2 + etp.^2));

        case {'trace','volumetric','dilatation'}
            C = err + ett + epp;

        case {'err','radial','drr','dradial','srr','sigmarr','taurr'}
            C = err;
        case {'ett','theta','dtt','dthetatheta','stt','sigmathetatheta','tauthetatheta'}
            C = ett;
        case {'epp','phi','dpp','dphiphi','spp','sigmaphiphi','tauphiphi'}
            C = epp;
        case {'ert','shear','rtheta','drt','dtr','drtheta','dthetar','srt','str','sigmartheta','sigmathetar','taurtheta'}
            C = ert;
        case {'erp','rphi','drp','dpr','drphi','dphir','srp','spr','sigmarphi','sigmaphir','taurphi'}
            C = erp;
        case {'etp','thetaphi','dtp','dpt','dthetaphi','dphitheta','stp','spt','sigmathetaphi','sigmaphitheta','tauthetaphi'}
            C = etp;
        otherwise
            error('Unknown StrainScalar/StrainRateScalar option: %s', char(scalarChoice));
    end
end

function opt = normalize_rate_derivative_options(opt, T, ep, R)
    if isempty(opt.Td)
        if ~isempty(opt.Tdot)
            opt.Td = opt.Tdot;
        elseif ~isempty(opt.V)
            opt.Td = opt.V;
        end
    end
    if isempty(opt.epd) && ~isempty(opt.epdot)
        opt.epd = opt.epdot;
    end
    if isempty(opt.Rd) && ~isempty(opt.Rdot)
        opt.Rd = opt.Rdot;
    end

    if ~isempty(opt.Td) && ~isequal(size(opt.Td), size(T))
        error('Td/Tdot/V must have the same size as T: [numel(N) x xN x Nt].');
    end
    if ~isempty(opt.epd) && ~isequal(size(opt.epd), size(ep))
        error('epd/epdot must have the same size as ep: [numel(N) x Nt].');
    end
    if ~isempty(opt.Rd) && numel(opt.Rd) ~= numel(R)
        error('Rd/Rdot must have the same number of entries as R.');
    end
    if ~isempty(opt.Rd)
        opt.Rd = opt.Rd(:);
    end
end

function opt = normalize_stress_options(opt, Nt)
    if isempty(opt.G) && ~isempty(opt.ShearModulus)
        opt.G = opt.ShearModulus;
    end
    if isempty(opt.mu) && ~isempty(opt.Viscosity)
        opt.mu = opt.Viscosity;
    end
    if isempty(opt.Pressure) && ~isempty(opt.P)
        opt.Pressure = opt.P;
    end
    if isempty(opt.alpha) && ~isempty(opt.StrainStiffeningAlpha)
        opt.alpha = opt.StrainStiffeningAlpha;
    elseif ~isempty(opt.StrainStiffeningAlpha)
        opt.alpha = opt.StrainStiffeningAlpha;
    end

    if is_stress_requested(opt)
        if isempty(opt.G) || isempty(opt.mu)
            error('Stress plotting requires both ''G'' and ''mu''. Optional: ''alpha'' and ''Pressure''/''P''.');
        end
        validate_frame_parameter(opt.G, Nt, 'G');
        validate_frame_parameter(opt.mu, Nt, 'mu');
        validate_frame_parameter(opt.alpha, Nt, 'alpha');
        if ~isempty(opt.Pressure)
            validate_frame_parameter(opt.Pressure, Nt, 'Pressure');
        end
    end
end

function validate_frame_parameter(v, Nt, name)
    if isempty(v)
        return;
    end
    if ~(isscalar(v) || numel(v) == Nt)
        error('%s must be a scalar or a vector with length Nt.', name);
    end
end

function val = value_at_frame(v, it, name)
    if isempty(v)
        error('%s is empty.', name);
    end
    if isscalar(v)
        val = v;
    else
        vv = v(:);
        val = vv(it);
    end
    val = real(val);
end

function tf = is_stress_requested(opt)
    sm = lower(strrep(char(opt.StrainMeasure), '-', '_'));
    tf = ismember(sm, {'stress','cauchy_stress','sigma','tau','constitutive_stress'});
end

function tf = is_rate_of_strain_requested(opt)
    sm = lower(strrep(char(opt.StrainMeasure), '-', '_'));
    tf = ismember(sm, {'rate','strain_rate','rate_of_strain','d','D','velocity_gradient_symmetric'});
end

function scalarChoice = active_tensor_scalar(opt)
    if is_stress_requested(opt) && ~isempty(opt.StressScalar)
        scalarChoice = opt.StressScalar;
    elseif is_rate_of_strain_requested(opt) && ~isempty(opt.StrainRateScalar)
        scalarChoice = opt.StrainRateScalar;
    else
        scalarChoice = opt.StrainScalar;
    end
end

function label = default_stress_label(scalarChoice)
    switch lower(char(scalarChoice))
        case {'err','radial','drr','dradial','srr','sigmarr','taurr'}
            label = '$\sigma_{rr}$';
        case {'ett','theta','dtt','dthetatheta','stt','sigmathetatheta','tauthetatheta'}
            label = '$\sigma_{\theta\theta}$';
        case {'epp','phi','dpp','dphiphi','spp','sigmaphiphi','tauphiphi'}
            label = '$\sigma_{\phi\phi}$';
        case {'ert','shear','rtheta','drt','dtr','drtheta','dthetar','srt','str','sigmartheta','sigmathetar','taurtheta'}
            label = '$\sigma_{r\theta}$';
        case {'erp','rphi','drp','dpr','drphi','dphir','srp','spr','sigmarphi','sigmaphir','taurphi'}
            label = '$\sigma_{r\phi}$';
        case {'etp','thetaphi','dtp','dpt','dthetaphi','dphitheta','stp','spt','sigmathetaphi','sigmaphitheta','tauthetaphi'}
            label = '$\sigma_{\theta\phi}$';
        case {'frobenius','norm','magnitude'}
            label = '$\|\sigma\|_F$';
        case {'trace','volumetric','dilatation'}
            label = '$\mathrm{tr}(\sigma)$';
        otherwise
            label = 'Cauchy stress';
    end
end

function label = default_rate_label(scalarChoice)
    switch lower(char(scalarChoice))
        case {'err','radial','drr','dradial','srr','sigmarr','taurr'}
            label = '$D_{rr}$';
        case {'ett','theta','dtt','dthetatheta','stt','sigmathetatheta','tauthetatheta'}
            label = '$D_{\theta\theta}$';
        case {'epp','phi','dpp','dphiphi','spp','sigmaphiphi','tauphiphi'}
            label = '$D_{\phi\phi}$';
        case {'ert','shear','rtheta','drt','dtr','drtheta','dthetar','srt','str','sigmartheta','sigmathetar','taurtheta'}
            label = '$D_{r\theta}$';
        case {'erp','rphi','drp','dpr','drphi','dphir','srp','spr','sigmarphi','sigmaphir','taurphi'}
            label = '$D_{r\phi}$';
        case {'etp','thetaphi','dtp','dpt','dthetaphi','dphitheta','stp','spt','sigmathetaphi','sigmaphitheta','tauthetaphi'}
            label = '$D_{\theta\phi}$';
        case {'frobenius','norm','magnitude'}
            label = '$\|D\|_F$';
        case {'trace','volumetric','dilatation'}
            label = '$\mathrm{tr}(D)$';
        otherwise
            label = 'Eulerian rate of strain';
    end
end

function C = principal_strain_3d(err, ett, epp, ert, erp, etp, mode)
    C = nan(size(err));
    for k = 1:numel(err)
        if ~isfinite(err(k)) || ~isfinite(ett(k)) || ~isfinite(epp(k)) || ...
           ~isfinite(ert(k)) || ~isfinite(erp(k)) || ~isfinite(etp(k))
            continue;
        end
        E = [err(k), ert(k), erp(k); ...
             ert(k), ett(k), etp(k); ...
             erp(k), etp(k), epp(k)];
        lam = eig(E);
        switch mode
            case 'signed_maxabs'
                [~,ii] = max(abs(lam));
                C(k) = lam(ii);
            case 'max'
                C(k) = max(lam);
            case 'min'
                C(k) = min(lam);
        end
    end
end

function rb = bubble_radius_general(theta, sx, epNow, Rcur, N, opt)
    az = slice_azimuth_from_side(sx, opt);
    rb = bubble_radius_eval(theta, az, epNow, Rcur, N, opt);
end

function rb = bubble_radius_eval(theta, az, epNow, Rcur, N, opt)
    rb = Rcur * ones(size(theta));
    for j = 1:numel(N)
        n = N(j);
        Y = harmonic_value_for_plot(n, theta, az, opt);
        rb = rb + Rcur * epNow(j) * Y;
    end
    rb = real(rb);
end

function [xBubble, zBubble] = bubble_curve_level_set(theta, sx, epNow, Rcur, N, opt)
    rb = bubble_radius_general(theta, sx, epNow, Rcur, N, opt);
    xBubble = sx .* rb .* sin(theta);
    zBubble =      rb .* cos(theta);
end

function [xBubble, zBubble] = display_boundary_curve(phiPlot, epNow, Rcur, N, opt)
    [~, ~, Xfun, Zfun, rb] = make_display_polar_grid(phiPlot, epNow, Rcur, N, opt);
    xBubble = Xfun(rb);
    zBubble = Zfun(rb);
end

function [theta, az, Xfun, Zfun, rb] = make_display_polar_grid(phiPlot, epNow, Rcur, N, opt)
    % Returns angular coordinates used to evaluate the harmonics and two
    % function handles mapping a radius array onto the displayed 2D plane.
    plane = resolve_slice_plane(opt);

    switch plane
        case 'meridional'
            theta = meridional_theta_from_plot_angle(phiPlot, opt.ThetaPoleEps);
            sx = sign_nonzero(cos(phiPlot));
            az = slice_azimuth_from_side(sx, opt);
            rb = bubble_radius_eval(theta, az, epNow, Rcur, N, opt);
            Xfun = @(rr) sx .* rr .* sin(theta);
            Zfun = @(rr)      rr .* cos(theta);

        case 'equatorial'
            % Equatorial plane: theta = pi/2 and the displayed polar angle is
            % the physical azimuth. This is the view that reveals sectoral m=n
            % lobes, since Y_n^n(pi/2,phi) varies as cos(n phi) or sin(n phi).
            theta = (pi/2) * ones(size(phiPlot));
            az = opt.SlicePhi + phiPlot;
            rb = bubble_radius_eval(theta, az, epNow, Rcur, N, opt);
            Xfun = @(rr) rr .* cos(phiPlot);
            Zfun = @(rr) rr .* sin(phiPlot);

        otherwise
            error('Unknown SlicePlane: %s.', plane);
    end
end


function nPhi = display_phi_count(opt)
    % Use the same angular discretization for the colour mesh and the bubble
    % fill/outline.  If these are different, the white bubble patch can mask
    % parts of the first coloured cells near a corrugated wall, producing
    % apparent gaps or jagged coloured slivers at the interface.
    if opt.ColorMaterialByStrain
        nPhi = max([round(opt.StrainDisplayPhi), round(opt.ThetaBubble), 3]);
    else
        nPhi = max(round(opt.ThetaBubble), 3);
    end
end

function vq = interp_radial_current(rEval, vals, rq, method, opt)
    % Interpolate modal radial coefficients to the plotting radius.
    %
    % The solver grid starts at the mean spherical wall r=R(t), while the
    % displayed wall is the perturbed surface r=R(t)*(1+eps*Y).  Therefore
    % parts of the displayed exterior can lie slightly below the first solver
    % node.  The main colour path normally avoids those queries by using
    % WallFieldEvaluationMode='mean_wall_extension'.  If raw perturbed-wall
    % evaluation is requested, ClampQueryBelowMeanWall=true keeps interpolation
    % at the first solver node; setting it false uses a first-order linear
    % extension below the first solver node.
    rv = rEval(:);
    vv = vals(:);
    rq0 = real(rq);

    r1 = rv(1);
    rN = rv(end);
    rqUse = min(rq0, rN);

    if opt.ClampQueryBelowMeanWall
        rqUse = max(rqUse, r1);
        vq = interp1(rv, vv, rqUse, method, 'extrap');
        return;
    end

    rqBase = max(rqUse, r1);
    vq = interp1(rv, vv, rqBase, method, 'extrap');

    below = rqUse < r1;
    if any(below(:)) && numel(rv) >= 2
        slope1 = (vv(2) - vv(1)) ./ (rv(2) - rv(1));
        vq(below) = vv(1) + slope1 .* (rqUse(below) - r1);
    end
end

function plane = resolve_slice_plane(opt)
    plane = lower(char(opt.SlicePlane));
    if strcmp(plane, 'auto')
        switch lower(char(opt.AngularMode))
            case {'m_equals_n','m=n','sectoral','sectoral_mn'}
                plane = 'equatorial';
            otherwise
                plane = 'meridional';
        end
    end
    if ~ismember(plane, {'meridional','equatorial'})
        error('SlicePlane must be ''auto'', ''meridional'', or ''equatorial''.');
    end
end

function th = meridional_theta_from_plot_angle(phiPlot, thetaPoleEps)
    if nargin < 2 || isempty(thetaPoleEps)
        thetaPoleEps = 0;
    end
    th = atan2(abs(cos(phiPlot)), sin(phiPlot));
    th = max(thetaPoleEps, min(pi - thetaPoleEps, th));
end

function az = slice_azimuth_from_side(sx, opt)
    % Right half of the meridional slice uses SlicePhi.
    % Left half uses SlicePhi + pi.
    az = opt.SlicePhi * ones(size(sx));
    az(sx < 0) = opt.SlicePhi + pi;
end

function Y = harmonic_value_for_plot(n, theta, az, opt)
    outputSize = size(theta);
    thetaRow = theta(:).';
    mode = lower(char(opt.AngularMode));
    switch mode
        case {'axisymmetric','axisym','m0','m_equals_0'}
            Y = Ynm0(n, thetaRow);

        case {'m_equals_n','m=n','sectoral','sectoral_mn'}
            m = n;
            x = cos(thetaRow);
            Pn = legendre(n, x);
            Pnm = squeeze(Pn(m+1,:));
            if m == 0
                normC = sqrt((2*n + 1)/(4*pi));
            else
                normC = sqrt(2) * sqrt((2*n + 1)/(4*pi) * ...
                    exp(gammaln(n-m+1) - gammaln(n+m+1)));
            end
            azRow = az(:).';
            switch lower(char(opt.RealHarmonicPart))
                case {'cos','cosine'}
                    trig = cos(m .* azRow);
                case {'sin','sine'}
                    trig = sin(m .* azRow);
                otherwise
                    error('RealHarmonicPart must be ''cos'' or ''sin''.');
            end
            Y = normC .* Pnm .* trig;

        otherwise
            error('Unknown AngularMode: %s. Use ''axisymmetric'' or ''m_equals_n''.', char(opt.AngularMode));
    end
    Y = reshape(Y, outputSize);
end

function [Y, Yt, Ytt, Yp, Ytp, Ypp] = harmonic_values_for_plot(n, theta, az, opt)
    mode = lower(char(opt.AngularMode));
    switch mode
        case {'axisymmetric','axisym','m0','m_equals_0'}
            Y   = Ynm0(n, theta);
            Yt  = dYnm0_dtheta(n, theta);
            Ytt = ddYnm0_dtheta2(n, theta);
            Yp  = zeros(size(Y));
            Ytp = zeros(size(Y));
            Ypp = zeros(size(Y));

        case {'m_equals_n','m=n','sectoral','sectoral_mn'}
            m = n;
            [Y, Yt, Ytt, Yp, Ytp, Ypp] = real_spherical_harmonic_derivs(n, m, theta, az, opt.RealHarmonicPart);

        otherwise
            error('Unknown AngularMode: %s. Use ''axisymmetric'' or ''m_equals_n''.', char(opt.AngularMode));
    end
end

function [Y, Yt, Ytt, Yp, Ytp, Ypp] = real_spherical_harmonic_derivs(n, m, theta, az, part)
    theta = theta(:).';
    az = az(:).';

    x = cos(theta);
    s = sin(theta);
    sSafe = max(abs(s), 1e-12);
    c = cos(theta);

    Pn_all = legendre(n, x);
    Pnm = squeeze(Pn_all(m+1,:));

    if n-1 >= m
        Pnm1_all = legendre(n-1, x);
        Pn1m = squeeze(Pnm1_all(m+1,:));
    else
        Pn1m = zeros(size(Pnm));
    end

    if m == 0
        normC = sqrt((2*n + 1)/(4*pi));
    else
        normC = sqrt(2) * sqrt((2*n + 1)/(4*pi) * exp(gammaln(n-m+1) - gammaln(n+m+1)));
    end

    dP_dtheta = (n .* x .* Pnm - (n + m) .* Pn1m) ./ sSafe;

    part = lower(char(part));
    switch part
        case {'cos','cosine'}
            trig = cos(m .* az);
            trig_p = -m .* sin(m .* az);
            trig_pp = -m^2 .* cos(m .* az);
        case {'sin','sine'}
            trig = sin(m .* az);
            trig_p = m .* cos(m .* az);
            trig_pp = -m^2 .* sin(m .* az);
        otherwise
            error('RealHarmonicPart must be ''cos'' or ''sin''.');
    end

    Y   = normC .* Pnm       .* trig;
    Yt  = normC .* dP_dtheta .* trig;
    Yp  = normC .* Pnm       .* trig_p;
    Ytp = normC .* dP_dtheta .* trig_p;
    Ypp = normC .* Pnm       .* trig_pp;

    % Spherical harmonic identity:
    % Y_tt + cot(theta)Y_t + Y_pp/sin(theta)^2 + n(n+1)Y = 0.
    cotTH = c ./ sSafe;
    Ytt = -cotTH .* Yt - Ypp ./ (sSafe.^2) - n*(n+1).*Y;

    Y(~isfinite(Y)) = 0;
    Yt(~isfinite(Yt)) = 0;
    Ytt(~isfinite(Ytt)) = 0;
    Yp(~isfinite(Yp)) = 0;
    Ytp(~isfinite(Ytp)) = 0;
    Ypp(~isfinite(Ypp)) = 0;
end

function s = sign_nonzero(x)
    s = sign(x);
    s(s == 0) = 1;
end

function y = sign_preserving_max(absx, floorVal)
    y = max(absx, floorVal);
end

function Y = Ynm0(n, theta)
    x = cos(theta(:).');
    P = legendre(n, x);
    P0 = squeeze(P(1,:));
    Y = sqrt((2*n + 1)/(4*pi)) .* P0;
end

function dY = dYnm0_dtheta(n, theta)
    theta = theta(:).';
    if n == 0
        dY = zeros(size(theta));
        return;
    end
    x = cos(theta);
    c = sqrt((2*n + 1)/(4*pi));
    Pn = legendre(n, x);
    Pn = squeeze(Pn(1,:));
    Pnm1 = legendre(n-1, x);
    Pnm1 = squeeze(Pnm1(1,:));
    s = sin(theta);
    dY = zeros(size(theta));
    interior = abs(s) > 1e-12;
    dY(interior) = c * n .* (x(interior).*Pn(interior) - Pnm1(interior)) ./ s(interior);
    dY(~isfinite(dY)) = 0;
end

function ddY = ddYnm0_dtheta2(n, theta)
    theta = theta(:).';
    Y = Ynm0(n, theta);
    dY = dYnm0_dtheta(n, theta);
    s = sin(theta);
    c = cos(theta);
    cotTH = c ./ max(abs(s), 1e-8);
    ddY = -cotTH .* dY - n*(n+1).*Y;
    ddY(~isfinite(ddY)) = 0;
end

function cmap = resolve_colormap(cmapIn)
    if isnumeric(cmapIn)
        cmap = cmapIn;
        return;
    end
    if isa(cmapIn, 'function_handle')
        cmap = cmapIn(256);
        return;
    end
    name = char(cmapIn);
    try
        cmap = feval(name, 256);
    catch
        warning('Could not resolve colormap %s; using parula.', name);
        cmap = parula(256);
    end
end

function scale = normalize_color_scale(scaleIn)
    scale = lower(strtrim(char(scaleIn)));
    switch scale
        case {'linear','lin'}
            scale = 'linear';
        case {'log','logarithmic','log10'}
            scale = 'log';
        otherwise
            error('Unknown cbScale: %s. Use ''linear'' or ''log''.', char(scaleIn));
    end
end

function tf = is_log_color_scale(opt)
    tf = strcmpi(char(opt.cbScale), 'log');
end

function vv = color_values_for_limits(C, opt)
    if is_log_color_scale(opt)
        vv = abs(C(isfinite(C)));
        vv = vv(vv > 0);
    else
        vv = C(isfinite(C));
    end
end

function cLim = default_color_limits(opt)
    if is_log_color_scale(opt)
        cLim = [1e-12 1];
    else
        cLim = [-1 1];
    end
end

function cLim = normalize_explicit_color_limits(cLimIn, opt)
    validateattributes(cLimIn, {'numeric'}, {'vector','numel',2,'finite'});
    cLim = real(cLimIn(:).');
    if is_log_color_scale(opt)
        cLim = unique(sort(abs(cLim(cLim ~= 0))));
        if isempty(cLim)
            cLim = default_color_limits(opt);
        elseif numel(cLim) == 1
            cLim = [max(cLim(1)*1e-6, realmin), cLim(1)];
        else
            cLim = [cLim(1), cLim(end)];
        end
    end
end

function C = prepare_color_data_for_scale(C, cLim, opt)
    if ~is_log_color_scale(opt)
        return;
    end
    C = abs(C);
    C(~isfinite(C) | C <= 0) = NaN;
    if numel(cLim) == 2 && isfinite(cLim(1)) && cLim(1) > 0
        C(C < cLim(1)) = cLim(1);
    end
end

function levels = contour_levels_for_scale(cLim, opt)
    nLevels = max(2, round(opt.ContourLevels));
    if is_log_color_scale(opt)
        levels = logspace(log10(cLim(1)), log10(cLim(2)), nLevels);
    else
        levels = nLevels;
    end
end

function apply_color_axis_scale(ax, opt)
    try
        ax.ColorScale = char(opt.cbScale);
    catch
        if is_log_color_scale(opt)
            warning('cbScale:UnsupportedColorScale', ...
                'This MATLAB version does not support logarithmic color scaling on this axes.');
        end
    end
end

function C = fillmissing_nearest_2d(C)
    if all(isfinite(C(:)))
        return;
    end
    [nr,nc] = size(C);
    [rr,cc] = ndgrid(1:nr, 1:nc);
    good = isfinite(C);
    if ~any(good(:))
        C(:) = 0;
        return;
    end
    try
        F = scatteredInterpolant(rr(good), cc(good), C(good), 'nearest', 'nearest');
        bad = ~good;
        C(bad) = F(rr(bad), cc(bad));
    catch
        % Fallback without scatteredInterpolant: row/column sweeps.
        for i = 1:nr
            row = C(i,:);
            g = isfinite(row);
            if any(g)
                row(~g) = interp1(find(g), row(g), find(~g), 'nearest', 'extrap');
                C(i,:) = row;
            end
        end
        for j = 1:nc
            col = C(:,j);
            g = isfinite(col);
            if any(g)
                col(~g) = interp1(find(g), col(g), find(~g), 'nearest', 'extrap');
                C(:,j) = col;
            end
        end
        C(~isfinite(C)) = 0;
    end
end

function pos = centered_normalized_position(w, h)
    w = min(max(real(w), 0.05), 1);
    h = min(max(real(h), 0.05), 1);
    pos = [(1-w)/2, (1-h)/2, w, h];
end

function s = sanitize_latex_label(s)
    if isstring(s)
        s = char(s);
    end
    if ~ischar(s)
        s = char(string(s));
    end
    s = strrep(s, '\\', '\');
end

function idx = nearest_time_index(t, tq)
    [~, idx] = min(abs(t - tq));
end
