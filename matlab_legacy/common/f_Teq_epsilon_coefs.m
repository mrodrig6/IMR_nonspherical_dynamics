function [epCoef, epdCoef] = f_Teq_epsilon_coefs(R, Rd, Rdd, r, n, Ca, alph, Re, ...
    rs, vs)
%F_TEQ_EPSILON_COEFS Coefficients multiplying epsilon and epsilon-dot in T rows.

r = r.'; rs = rs.';

epCoef = -(2.*(2+n).*r.^(-12-n).*R.^(3+n).*(3.*r.^4.*R.*((7+n).*R.^3.*Rd.^2-r.^3.*((5+n).*Rd.^2+...
    R.*Rdd)+(10.*(3+n).*r.*R.*Rd)./Re).*rs.^7+(vs.*(-((-1+3.*alph).*r.^4.*rs.^2.*(5.*(3+n).*r.^6-...
    (37+7.*n).*r.^3.*vs+3.*(7+n).*vs.^2))+alph.*(15.*(3+n).*r.^12+(-177+n+4.*n.^2).*r.^9.*vs-...
    8.*(-36+n.*(3+n)).*r.^6.*vs.^2+(-222+n.*(21+5.*n)).*r.^3.*vs.^3-(-6+n).*(11+n).*vs.^4)))./Ca))./((1+n).*rs.^7);

epdCoef = (6.*(2+n).*r.^(-5-n).*R^(5+n).*Rd)./(1+n);

end
