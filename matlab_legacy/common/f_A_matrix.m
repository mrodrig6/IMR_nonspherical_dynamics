function A = f_A_matrix(r, R, Rd, Rdd, Req, N, xN, H1, H2, H3, w, Re, ...
                Ca, We, alph,  a, dxdr, dxdr2, dxdr21, dxdr3, dxdr32, ... 
                dxdr31, dr, oneW, W, rs, vs, sr, forcedep, blockSize, mod, rot)
%F_A_MATRIX Assemble the linear operator for the coupled modal evolution.


nmode = numel(N);
A = zeros(nmode*blockSize);

if mod == "me" && rot == "rot"
    I = eye(xN);
    h = 2/(xN-1);
    Aadv = f_advective_D1_operator(a, h);
else
    I = [];
    Aadv = [];
end

for i = 1:nmode
    n = N(i);

    % create state vector
    [idxT, idxV, idxep, idxepd, ~] = f_get_indicies(forcedep, mod, xN, i, rot);

    if mod == "me" && rot == "rot"
        % setup general system
        % idxV now stores W = dT/dt|_r, not dT/dt|_x
        A(idxT, idxT) = A(idxT, idxT) + Aadv;
        A(idxT, idxV) = I;
    end

    if forcedep ~= 'T'
        A(idxep, idxepd) = 1;
    end

    if mod == "me"
        if forcedep ~= 'T'
            [xi, eta] = f_epsilon_coeffs(R, Rd, sr, Rdd, n, Req, We, Re, Ca, alph, mod);
            A(idxepd, idxep) = A(idxepd, idxep) - xi;
            A(idxepd, idxepd) = A(idxepd, idxepd) - eta;
            
            if rot == "rot"
                [epCoef, epdCoef] = f_Teq_epsilon_coefs(R, Rd, Rdd, r, n, Ca, ...
                    alph, Re, rs, vs);
                A(idxV, idxep) = A(idxV, idxep) - epCoef;
                A(idxV, idxepd) = A(idxV, idxepd) - epdCoef;
            end
        end
        
        if rot == "rot"
            % start by computing all terms that do not depend on integrals
            [Ct1, Ct2, Ct3, Ct4, Ct5, Ct6, Ct7] = f_non_int_coeffs(r,...
                R, Rd, Rdd, n, Ca, Re, alph, rs, vs);

            A = f_mod_A_nonint_terms(A, Ct1, Ct2, Ct3, Ct4, Ct5, Ct6, Ct7, ...
                dxdr, dxdr2, dxdr21, dxdr3, dxdr32, dxdr31, H1, ...
                H2, H3, idxT, idxV);

            % Now to add the complicated terms, kappa, mathcal (J and K)
            % Account first for the contribution to the soltion from kappa in the
            % bulk
            [TCoefkappa, VCoefkappa] = f_bulk_kappa(r, R, Rd, Rdd, n, Ca, Re, alph, ...
                w, dr, rs, vs);

            A(idxV, idxT) = A(idxV, idxT) - TCoefkappa;
            A(idxV, idxV) = A(idxV, idxV) - VCoefkappa;

            % now account for A and S sub \mathcal J and \mathcal K
            [TCoefKJ, VCoefKJ] = f_bulk_KJ(r, rs, vs, dr, oneW, W, n, R, Rd, Rdd,...
                Ca, Re, alph);
            A(idxV, idxV) = A(idxV, idxV) - VCoefKJ;
            A(idxV, idxT) = A(idxV, idxT) - TCoefKJ;
        end
       
    elseif mod == "Pros"
        if rot == "rot"
            diagCoef = 1/Re*n*(n+1)./r.^2;
            colCoef = Rd*R^2./r.^2;

            A(idxT, idxT) = A(idxT, idxT) + ...
                1/Re*(dxdr21(:).*H1) + ...
                1/Re*(dxdr2(:).*H2) - ...
                (dxdr(:).*H1).*colCoef(:).';

            diagIdx = sub2ind(size(A), idxT(:), idxT(:));
            A(diagIdx) = A(diagIdx) - diagCoef(:);
        end
        if forcedep ~= 'T'
            [xi, eta] = f_epsilon_coeffs(R, Rd, sr, Rdd, n, Req, We, Re, Ca, alph, mod);
            A(idxepd, idxep) = A(idxepd, idxep) - xi;
            A(idxepd, idxepd) = A(idxepd, idxepd) - eta;
        end
    end

end
end
