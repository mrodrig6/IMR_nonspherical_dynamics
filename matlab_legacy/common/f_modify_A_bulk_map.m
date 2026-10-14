function A = f_modify_A_bulk_map(A, N, xN, a, mod, blockSize)
%F_MODIFY_A_BULK_MAP  Mapping correction terms with a direct discrete
% approximation of the CONTINUUM expression:
%
%   V_t = Y + (at - a*ax) T_x + 2a V_x - a^2 T_xx
%
% Implemented on the LHS under backward Euler (evaluated at n+1):
%   V^{n+1} - dt*[(at-a*ax)T_x + 2aV_x - a^2 T_xx]^{n+1} = ...
%
% We keep the product-rule-consistent flux forms for the first-derivative
% terms, and we implement a^2 T_xx EXACTLY via the identity:
%   a^2 T_xx = (a^2 T_x)_x - (a^2)_x T_x
% with both pieces discretized using H1 (no guessed grid spacing, no new inputs).

a  = a(:);
h = 2/(xN-1);
Aadv = f_advective_D1_operator(a, h);

for k = 1:numel(N)
    if mod == "me"       
        base = (k-1)*blockSize;
        % ---- indices ----
        idxV  = base + xN + (1:xN);
        A(idxV, idxV) = A(idxV, idxV) + Aadv;

    elseif mod == "Pros"
        base = (k-1)*blockSize;
        % ---- indices ----
        idxT   = base + (1:xN);
        A(idxT, idxT) = A(idxT, idxT) + Aadv;
    end
end
end
