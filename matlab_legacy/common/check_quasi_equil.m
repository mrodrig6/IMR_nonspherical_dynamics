function [isEquil, eqHits, relChangeMax, pinc] = check_quasi_equil(r, ...
    ti, t, P, T_tot, Td_tot, nModes, xN, eqTol, nConsec, eqHits, istep)
%CHECK_QUASI_EQUIL  Detect periodic steady state (quasi-equilibrium) for periodic forcing
%
% This version treats "quasi-equilibrium" as convergence to a PERIODIC ORBIT:
% compare the solution at the same phase each cycle (stroboscopic sampling).
%
% Criterion:
%   relChange(mode) = ||T_k - T_{k-1}|| / (||T_k|| + eps)
%   relChangeMax    = max over modes
%   if relChangeMax < eqTol for nConsec consecutive cycles -> isEquil = true
%
% Also produces a persistent diagnostic figure showing:
%   (1) relChangeMax and relChange(mode1) vs time
%   (2) stroboscopic probe values at one x index (current vs previous cycle)

    isEquil = false;
    relChangeMax = NaN;
    pinc = 0;

    % Need at least one full period history
    if ti < 25*P
        eqHits = 0;
        return
    end

    % --------------------------
    % Choose stroboscopic times:
    % tk   = latest time <= ti
    % tk-1 = tk - P
    % Use nearest indices in t-grid
    % --------------------------
    tk = ti;
    tkm1 = ti - P;
    

    [~, ik]   = min(abs(t - tk));
    [~, ikm1] = min(abs(t - tkm1));

    % Must have valid ordering and enough data
    if ikm1 < 1 || ik < 1 || ikm1 == ik
        eqHits = 0;
        return
    end

    % Extract snapshots: nModes-by-xN
    
    pinc = ik-ikm1;

    ikm2 = ik - 2*pinc;
    Tk   = squeeze(T_tot(:, :, ik));
    Tkm2 = squeeze(T_tot(:, :, ikm2));

    % --------------------------
    % Relative change per mode
    % Use RMS norm over x for each mode
    % --------------------------
    relChange = zeros(nModes,1);
    for m = 1:nModes
        a = Tk(m,1:length(r(r<5)));
        b = Tkm2(m,1:length(r(r<5)));
        num = sqrt(mean((a - b).^2));
        den = sqrt(mean(a.^2)) + 1e-14;
        relChange(m) = num / den;
    end
    
    relChangeMax = max(relChange);

    % --------------------------
    % Hit counter logic
    % --------------------------
    if (relChangeMax < eqTol)
        eqHits = eqHits + 1;
    else
        eqHits = 0;
    end

    if eqHits >= nConsec
        fprintf('Periodic steady-state detected at step %d, t = %.6g\n', istep, ti);
        isEquil = true;
    end

    % % ==========================================================
    % % PERSISTENT DIAGNOSTIC PLOTS (no nexttile; robust)
    % % ==========================================================
    % persistent figH axTop axBot hMax hM1 hProbeNow hProbePrev ...
    %            tHist dMaxHist dM1Hist probeNowHist probePrevHist
    % 
    % % initialize history buffers
    % if isempty(tHist)
    %     tHist = [];
    %     dMaxHist = [];
    %     dM1Hist  = [];
    %     probeNowHist  = [];
    %     probePrevHist = [];
    % end
    % 
    % % Append history every call (or you can thin it if you want)
    % tHist(end+1,1)    = ti;
    % dMaxHist(end+1,1) = relChangeMax;
    % dM1Hist(end+1,1)  = relChange(1);   % mode 1 diagnostic
    % 
    % % probe index (fixed, interior)
    % iprobe = 1;
    % probeNow  = Tk(1, iprobe);
    % %probePrev = Tkm1(1, iprobe);
    % 
    % % Create figure once
    % if isempty(figH) || ~isvalid(figH)
    %     figH = figure('Color','w','Name','Quasi-equilibrium diagnostics (stroboscopic)');
    %     axTop = subplot(2,1,1,'Parent',figH); hold(axTop,'on'); grid(axTop,'on'); box(axTop,'on');
    %     set(axTop,'YScale','log');   % <-- force log scaling
    %    % title(axTop,'Stroboscopic period-to-period relative change', 'Interpreter','latex');
    %     xlabel(axTop,'$t$ (as passed in)','Interpreter','latex');
    %     ylabel(axTop,'$\delta$','Interpreter','latex');
    % 
    %     % precreate line objects
    %     hMax = semilogy(axTop, nan, nan, 'k-',  'LineWidth', 1.8);
    % end
    % 
    % % Update plots
    % if isvalid(figH)
    %     set(hMax, 'XData', tHist, 'YData', dMaxHist);
    % 
    %     % Optional: show tolerance line on top plot
    %     % (draw once by checking for an existing line is more work; keep simple)
    %     yline(axTop, eqTol, ':', 'Interpreter','latex');
    % 
    %     drawnow limitrate
    % end
end


