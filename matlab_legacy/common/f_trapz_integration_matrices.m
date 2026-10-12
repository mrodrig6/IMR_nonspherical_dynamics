function [W, w, One_wT] = f_trapz_integration_matrices(x)
%COMPUTE_TRAPZ_INTEGRATION_MATRICES Trapezoidal quadrature weights and
% cumulative integration matrix on a (possibly nonuniform) 1D grid.
%
% Inputs:
%   x : grid vector (length xN), strictly increasing
%
% Outputs:
%   w      : xN×1 trapezoidal weights for ∫ f dx ≈ w' f
%   W      : xN×xN lower-tri matrix for cumulative integrals:
%            F_i = ∫_{x1}^{xi} f(s) ds ≈ (W f)_i
%   One_wT : xN×xN matrix ones(xN,1)*w'

    x = x(:);                 % force column
    xN = numel(x);

    if xN < 2
        error('Need at least 2 grid points.');
    end
    if any(~isfinite(x))
        error('x must be finite.');
    end

    dx = diff(x);
    if any(dx <= 0)
        error('x must be strictly increasing.');
    end

    % ---- trapezoidal weights w ----
    w = zeros(xN,1);
    w(1)   = dx(1)/2;
    if xN > 2
        w(2:xN-1) = (dx(1:end-1) + dx(2:end))/2;
    end
    w(end) = dx(end)/2;

    % ---- cumulative trapezoidal integration matrix W ----
    W = zeros(xN,xN);
    % row 1 already zeros: integral to x1 is 0

    if xN == 2
        % F2 = ∫_{x1}^{x2} f ≈ dx1/2*(f1+f2)
        W(2,1) = dx(1)/2;
        W(2,2) = dx(1)/2;
    else
        for i = 2:xN
            % Contribution from panel endpoints:
            % F_i uses panels 1..i-1 with trapezoidal weights restricted to [x1, xi]
            W(i,1) = dx(1)/2;           % f1
            for k = 2:(i-1)
                W(i,k) = (dx(k-1)+dx(k))/2;  % interior nodes
            end
            W(i,i) = dx(i-1)/2;         % f_i
        end
    end

    One_wT = ones(xN,1) * (w.');
end
