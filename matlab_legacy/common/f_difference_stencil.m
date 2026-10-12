function D = f_difference_stencil(N, deriv_order, h)
%COMPUTE_DIFFERENCE_STENCIL  N×N finite-difference derivative matrix (2nd-order accurate)
%
% Builds an N×N sparse matrix D such that D*f approximates the
% deriv_order-th derivative of f on a uniform grid with spacing h.
%  - deriv_order = 1: centered interior, 2nd-order one-sided boundaries
%  - deriv_order = 2: centered interior, 2nd-order one-sided boundaries
%  - deriv_order = 3: CONSISTENT composition D3 = D1*D2 using the SAME D1,D2

    if nargin < 3 || isempty(h)
        h = 1;
    end

    validateattributes(N, {'numeric'}, {'scalar','integer','>=',1});
    validateattributes(deriv_order, {'numeric'}, {'scalar','integer'});
    validateattributes(h, {'numeric'}, {'scalar','real','finite','nonzero'});

    n = N;

    switch deriv_order
        case 1
            if n < 3
                error('For 1st derivative, need N >= 3.');
            end
            e = ones(n,1);
            D = spdiags([-e, e], [-1, 1], n, n) / (2*h);

            D(1,:) = 0;      D(1,1:3) = [-3, 4, -1] / (2*h);
            D(n,:) = 0;      D(n,n-2:n) = [1, -4, 3] / (2*h);

        case 2
            if n < 4
                error('For 2nd derivative, need N >= 4.');
            end
            e = ones(n,1);
            D = spdiags([e, -2*e, e], [-1, 0, 1], n, n) / (h^2);

            D(1,:) = 0;      D(1,1:4) = [2, -5, 4, -1] / (h^2);
            D(n,:) = 0;      D(n,n-3:n) = [-1, 4, -5, 2] / (h^2);

        case 3
            if n < 5
                error('For 3rd derivative, need N >= 5.');
            end

            % Build D1 and D2 using THIS SAME function to guarantee consistency
            D1 = f_difference_stencil(n, 1, h);
            D2 = f_difference_stencil(n, 2, h);

            D  = D1 * D2;

        otherwise
            error('Supported deriv_order: 1, 2, 3.');
    end

    D = sparse(D);
end