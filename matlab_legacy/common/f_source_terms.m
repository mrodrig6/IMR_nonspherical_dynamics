function s = f_source_terms(ep, epd, R, Rd, Rdd, Ro, r, N, xN, Ca, alph, Re, ...
    rs, vs, blockSize, forcedep, mod, rot)
%F_SOURCE_TERMS Build the T-field source vector for prescribed epsilon history.

% s: vector of length length(N)*2*xN for the source terms
% r: physical coordinate space
% Ap: Source terms to the vorticity equation from the acceleration of the
% potential model
% Sp: Source terms to the vorticity equation from the stress of the
% potential model

s = zeros(length(N)*blockSize,1);

sr = R/Ro;

% Compute source terms to PDE when epsilon is not part of the state vector
for i = 1:length(N)

    [~, idxV, ~, ~, ~] = f_get_indicies(forcedep, mod, xN, i, rot);
    n = N(i);
    
    if rot == "rot"
    % ----------- Source Terms for T equation --------------------------- %

    Ap = 6*(n+2)/(n+1)*R^(n+4)./(r.^(n+9)).*(ep(i).*(Rd^2.*((n+5).*r.^3 ...
        -(n+7).*R^3)+r.^3.*R.*Rdd)+r.^3.*R.*Rd.*epd(i));
    Sp = 2*ep(i)*(n+2)./(n+1)*R.^(n+3)./(rs.^7.*r.^(n+13)).*((30.*(3+n).*r.^5.*Rd.*R^2.*rs.^7.)/Re+ ...
        1/Ca.*vs.*(-((-1+3.*alph).*r.^4.*rs.^2.*(5.*(3+n).*r.^6-(37+7.*n).*r.^3.*vs+3.*(7+n).*vs.^2))+ ...
        alph.*(15.*(3+n).*r.^12+(-177+n+4.*n.^2).*r.^9.*vs-8.*(-36+n.*(3+n)).*r.^6.*vs.^2+(-222+n.*(21+5.*n)).*r.^3.*vs.^3-(-6+n).*(11+n).*vs.^4)));

    s(idxV) = Sp - Ap;
    end

end
end
