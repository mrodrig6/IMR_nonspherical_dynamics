function [Ct1, Ct2, Ct3, Ct4, Ct5, Ct6, Ct7] = f_non_int_coeffs(r, R, Rd, Rdd, n, Ca, Re, alph, rs, vs)
%F_NON_INT_COEFFS Compute local coefficients in the rotational T equation.

Ct1 = -((R.^2.*Rd)./(r.^2.*Re));

Ct2 = -(1./Re);

Ct3 = (r.^4.*R.^2.*Rd.*(R.^2.*Rd+(8.*r)./Re)+((-1+3.*alph).*r.^4.*rs.^4)./Ca-...
    (alph.*rs.^2.*(3.*r.^6-2.*r.^3.*vs+vs.^2))./Ca)./r.^8;

Ct4 = (2.*R.^2.*Rd)./r.^2;

Ct5 = (r.^4.*R.*(r.^3.*R.*Rdd+Rd.*(2.*(r.^3-4.*R.^3).*Rd+((-36+n+n.^2).*r.*R)./Re))- ...
    (8.*vs.*(3.*alph.*r.^6+(1-3.*alph).*r.^4.*rs.^2-4.*alph.*r.^3.*vs+2.*alph.*vs.^2))./(Ca.*rs))./r.^9;

Ct6 = (-6.*R.^2.*Rd+(n.*(1+n).*r)./Re)./r.^3;

Ct7 = (2.*r.^4.*R.*(-2.*r.^3.*R.*Rdd+Rd.*((-4.*r.^3+13.*R.^3).*Rd+((48+n+n.^2).*r.*R)./Re)).*rs.^4-...
    ((-1+3.*alph).*r.^4.*rs.^2.*(n.*(1+n).*r.^6+20.*r.^3.*vs-26.*vs.^2))./Ca+...
    (alph.*(3.*n.*(1+n).*r.^12-2.*(-5+n).*(6+n).*r.^9.*vs+(-190+9.*n.*(1+n)).*r.^6.*vs.^2- ...
    8.*(-27+n+n.^2).*r.^3.*vs.^3+2.*(-43+n+n.^2).*vs.^4))./Ca)./(r.^10.*rs.^4);
end
