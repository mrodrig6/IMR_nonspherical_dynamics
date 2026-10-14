function A = f_mod_A_nonint_terms(A, Ct1, Ct2, Ct3, Ct4, Ct5, Ct6, Ct7, dxdr, ...
    dxdr2, dxdr21, dxdr3, dxdr32, dxdr31, H1, ...
    H2, H3, idxT, idxV)

% T directly Ct7
A = subtract_block_diagonal(A, idxV, idxT, Ct7);

% first r derivative of T, Ct5
A(idxV, idxT) = A(idxV, idxT) - (Ct5(:).*dxdr(:)).*H1;

% second spatial derivative of T, Ct3
% ========================================================================
% \partial_r^2 T = dxdr2*\partial_x^2 T + dxdr21*\partial_x T
% ========================================================================
dr2_coef = Ct3;
A(idxV, idxT) = A(idxV, idxT) - (dr2_coef(:).*dxdr2(:)).*H2;
A(idxV, idxT) = A(idxV, idxT) - (dr2_coef(:).*dxdr21(:)).*H1;

% third derivative of T, Ct1
% ========================================================================
% \partial_r^3 T = dxdr3*\partial_x^3 T + dxdr32*\partial_x^2 T + dxdr31*\partial_x T
% ========================================================================
dr3_coef = Ct1;
A(idxV, idxT) = A(idxV, idxT) - (dr3_coef(:).*dxdr3(:)).*H3;
A(idxV, idxT) = A(idxV, idxT) - (dr3_coef(:).*dxdr32(:)).*H2;
A(idxV, idxT) = A(idxV, idxT) - (dr3_coef(:).*dxdr31(:)).*H1;

% terms involving \partial_t T Ct6
% ========================================================================
% \partial_t T|r = V 
% ========================================================================
V_coeff = Ct6;
A = subtract_block_diagonal(A, idxV, idxV, V_coeff);

% first mixed derivative of T, Ct4
% ========================================================================
% \partial_t\partial_r T|r = \partial_r (V)
%                          = dxdr \partial_x*(V )
% ========================================================================
md1 = Ct4;
A(idxV, idxV) = A(idxV, idxV) - (md1(:).*dxdr(:)).*H1;

% second mixed derivative of T, Ct2
% ========================================================================
% \partial_t\partial_r^2 T|r = \partial_r^2 (V )
%                          = dxdr2 \partial_x^2*(V )
%                           + dxdr21 \partial_x*(V )
% ========================================================================
md2 = Ct2;
A(idxV, idxV) = A(idxV, idxV) - (md2(:).*dxdr2(:)).*H2;
A(idxV, idxV) = A(idxV, idxV) - (md2(:).*dxdr21(:)).*H1;

end

function A = subtract_block_diagonal(A, rowIdx, colIdx, values)
linIdx = sub2ind(size(A), rowIdx(:), colIdx(:));
A(linIdx) = A(linIdx) - values(:);
end
