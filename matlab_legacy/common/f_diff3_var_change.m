function [dxdr3_coef, dxdr2_coef, dxdr_coef] = f_diff3_var_change(x, R, L)
%F_DIFF3_VAR_CHANGE Chain-rule coefficients for third radial derivatives.

dxdr_coef = 3*(1-x).^4./(4*R^3*L^3);
dxdr2_coef = -3.*(1-x).^5./(4*R^3*L^3);
dxdr3_coef = (1-x).^6./(8*R^3*L^3);

end
