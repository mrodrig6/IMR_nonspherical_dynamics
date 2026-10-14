function Aadv = f_advective_D1_operator(c, h)
%F_ADVECTIVE_D1_OPERATOR  Matrix for RHS transport term c(x)*d/dx.
%
%   Aadv = f_advective_D1_operator(c, h)
%
% This function is for terms used on the RIGHT-HAND SIDE:
%
%       q_t = ... + c(x) q_x
%
% so Aadv*q approximates c(x).*q_x with upwinding chosen for that RHS form.
%
% IMPORTANT:
%   - If c > 0, characteristics move to smaller x, so use FORWARD bias.
%   - If c < 0, characteristics move to larger  x, so use BACKWARD bias.
%
% Inputs
%   c : scalar or N-vector coefficient multiplying q_x
%   h : uniform grid spacing (positive)
%
% Output
%   Aadv : N-by-N sparse matrix such that
%          Aadv*q approximates c(x).*q_x
%
% Notes
%   Uses split upwind:
%       c q_x ≈ c^+ Df q + c^- Db q
%   where
%       c^+ = max(c,0), c^- = min(c,0)
%       Df  = forward derivative  (used when c > 0)
%       Db  = backward derivative (used when c < 0)
%
% This is ONLY for first-derivative transport terms on the RHS.
% Do not use it for H2, H3, or products like H1*diag(a)*H1.

    if nargin < 2 || isempty(h)
        h = 1;
    end

    c = c(:);
    N = length(c);

    validateattributes(h, {'numeric'}, {'scalar','real','finite','positive'});

    cp = max(c, 0);
    cm = min(c, 0);

    Df = build_D1_forward(N, h);   % use when c > 0
    Db = build_D1_backward(N, h);  % use when c < 0

    Aadv = spdiags(cp, 0, N, N) * Df + spdiags(cm, 0, N, N) * Db;
    Aadv = sparse(Aadv);
end


% ========================================================================
% Local helpers
% ========================================================================

function D = build_D1_backward(n, h)
% Backward derivative in the interior:
%   f_x ~ (3f_i - 4f_{i-1} + f_{i-2})/(2h)
    D = spalloc(n, n, 3*n);

    if n < 3
        error('Need N >= 3 for 1st derivative.');
    elseif n == 3
        D(1,1:3) = [-3, 4, -1] / (2*h);
        D(2,1:3) = [-1, 0, 1]  / (2*h);
        D(3,1:3) = [ 1,-4, 3]  / (2*h);
        D = sparse(D);
        return;
    end

    % Left boundary closures
    D(1,1:3) = [-3, 4, -1] / (2*h);
    D(2,1:3) = [-1, 0, 1]  / (2*h);

    % Interior and right side
    for i = 3:n
        D(i,i-2:i) = [1, -4, 3] / (2*h);
    end

    D = sparse(D);
end

function D = build_D1_forward(n, h)
% Forward derivative in the interior:
%   f_x ~ (-3f_i + 4f_{i+1} - f_{i+2})/(2h)
    D = spalloc(n, n, 3*n);

    if n < 3
        error('Need N >= 3 for 1st derivative.');
    elseif n == 3
        D(1,1:3) = [-3, 4, -1] / (2*h);
        D(2,1:3) = [-1, 0, 1]  / (2*h);
        D(3,1:3) = [ 1,-4, 3]  / (2*h);
        D = sparse(D);
        return;
    end

    % Interior and left side
    for i = 1:n-2
        D(i,i:i+2) = [-3, 4, -1] / (2*h);
    end

    % Right boundary closures
    D(n-1,n-2:n) = [-1, 0, 1]  / (2*h);
    D(n,  n-2:n) = [ 1, -4, 3] / (2*h);

    D = sparse(D);
end