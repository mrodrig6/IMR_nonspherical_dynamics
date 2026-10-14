function [A] = f_ep_source_terms(r, xN, N, A, R, Rd, Rdd, Req, H1, dxdr, dr, w, ...
                        W, One_wT, Re, Ca, alph, vs, rs, sr, mod, forcedep, rot)
%F_EP_SOURCE_TERMS Add T-field source terms to the epsilon evolution rows.

nmodes = length(N);

for i = 1:nmodes
    n = N(i);
    [idxT, idxV, idxep, idxepd, ~] = f_get_indicies(forcedep, mod, xN, i, rot); 
    if mod == "me"  
        whole_block = idxT(1):idxepd;

% ======================================================================
        % Terms from fTintegrand
% ======================================================================
        fTPI = -((n.*(1+n).^2.*(-r.^4.*Rd.*(R.^4.*Rd+(4.*r.*R.^2)./Re)+((1- ...
            3.*alph).*r.^4.*rs.^4)./Ca+(3.*alph.*rs.^8)./Ca))./(r.^10.*R.^2));

        fVPI = -((n.*(1 + n)^2)./(r.^2.*R^2.*Re));
        fdTdrPI = -((n.*(1+n)^2.*Rd)./(r.^4.*Re));

        row_fT_T = (w(:).'.*dr.*fTPI) ...
            + (w(:).'.*dr.*fdTdrPI.*dxdr) * H1;

        row_fT_V =  w(:).'.*dr.*fVPI;

        A(idxepd, idxT) = A(idxepd, idxT) + row_fT_T;
        A(idxepd, idxV) = A(idxepd, idxV) + row_fT_V;

        % fTmod = w(:).'.*(fTPI + fdTdrPI.*dxdr*H1).*dr;
        % fVmod = w(:).'.*fVPI.*dr;
        % 
        % A(idxepd, idxV) = A(idxepd, idxV) + fVmod;
        % A(idxepd, idxT) = A(idxepd, idxT) + fTmod;

    
        [KmT, JmT] = f_J_K_kernel(r, dr, One_wT, W, n);
        [wkap] = f_kappa_kernel(r, dr, n, w);

        Li = A(idxV, whole_block);

% ======================================================================
        % Terms from kappa (all from FRHSFinal, GRHSFinal, HRHSFinal, FTRSfinal)
% ======================================================================
        
        % Modification from H
        Hkapddot = -n*R^(n-2);
        Hkapddotmod = Hkapddot.*wkap*Li;
        A(idxepd, whole_block) = A(idxepd, whole_block) + Hkapddotmod; 

        % Modification from G
        % simplification of f_T with H and G leads to one term for kapdot
        % already accounted for the leibniz term
        Gkapdot = (R^(-4+n).*(-2*n.*(1+n).*(4+n).*(1+2.*n)./Re-2.*n.*(2+n.*(6+n)).*R.*Rd))/((4+n));
        Gkapmod = Gkapdot.*wkap;
        A(idxepd, idxV) = A(idxepd, idxV) + (Gkapmod);

        % Modification from F
        Fkap = n.*(-(((4+n.*(8+n)).*R.^(-4+n).*((-1+n).*Rd.^2+R.*Rdd))./(4+n))-...
            2.*(1+n).*(1+2.*n).*(R/sr).^(-5+n).*sr.^(-12+n).*(((-2+n).*Rd.*sr.^7)./Re+...
            ((R/sr).*(sr.^4+alph.*(2-3.*sr.^4+sr.^6)))./Ca)+1./((6+n).*(9+n).*(12+...
            n).*R.^2).*(1+n).*(R/sr).^(-3+n).*sr.^(-11+n).*(-((6.*n.*(2+n).*(9+n).*(12+...
            n).*Rd.*sr.^8)./Re)+1./Ca.*(R/sr).*(-1+sr.^3).*((-1+3.*alph).*(13+n).*sr.^3.*(-n.*(2+...
            n).*(7+n)+n.*(86+n.*(35+3.*n)).*sr.^3+2.*(6+n).*(9+n).*sr.^6).*hyp2f1(-(1./3),4+n./3,5+n./3,1-1./sr.^3)+...
            2.*alph.*(n.*(2+n).*(11+n).*(39+n.*(13+n))-n.*(1398+n.*(1411+n.*(427+2.*n.*(25+n)))).*sr.^3+...
            (1+n).*(6+n).*(9+n).*(6+n.*(6+n)).*sr.^6+2.*(1+n).*(2+n).*(6+n).*(9+n).*sr.^9+4.*(6+n).*(9+...
            n).*sr.^12).*hyp2f1(1./3,4+n./3,5+n./3,1-1./sr.^3)+(-1+3.*alph).*(n.*(2+n).*(7+n).*(10+n)-...
            n.*(2+n).*(7+n).*(19+2.*n).*sr.^3+n.*(1+n).*(6+n).*(9+n).*sr.^6-2.*(6+n).*(9+...
            n).*sr.^9).*hyp2f1(2./3,4+n./3,5+n./3,1-1./sr.^3)-1./sr.^2.*alph.*(14+n).*(n.*(318+...
            n.*(175+n.*(33+2.*n)))-(-324+n.*(120+n.*(229+n.*(59+4.*n)))).*sr.^3+2.*(1+n).*(2+...
            n).*(6+n).*(9+n).*sr.^6+4.*(6+n).*(9+n).*sr.^9).*hyp2f1(1,(17+n)./3,5+n./3,1-1./sr.^3))));

        fkapmod = Fkap.*wkap;
        A(idxepd, idxT) = A(idxepd, idxT) + (fkapmod);

        % After simplifying f_T, G and H, one boundary term
        fT = -((n.*(1+n).*Rd.*(R.*Rd+(2.*(1+n))/Re))/R^4);
        A(idxepd, idxT(1)) = A(idxepd, idxT(1)) + fT;

% ======================================================================
        % Terms from f_J (all from fJintegrandfinal)
% ======================================================================
        % first T(r,t)

        fJT = 1./(1+2.*n).*n.*(1+n).^2.*r.^(-12-n).*((n.*r.^4.*(2.*r.^3-(7+n).*R.^3).*Rd.^2)./R+...
            n.*r.^7.*Rdd-(6.*n.*(2+n).*r.^5.*Rd)./Re-1./(Ca.*rs.^7.*sr.^2).*Req.*(-1+sr.^3).*(-((-1+...
            3.*alph).*r.^4.*rs.^2.*(2.*(2+n).*(3+n).*r.^6-(8+n.*(17+3.*n)).*r.^3.*vs+n.*(7+n).*vs.^2))+...
            alph.*(2.*(2+n).*(21+5.*n).*r.^12-(172+27.*n.*(7+n)).*r.^9.*vs+(152+n.*(235+29.*n)).*r.^6.*vs.^2-...
            3.*(16+n.*(47+5.*n)).*r.^3.*vs.^3+3.*n.*(11+n).*vs.^4)));
        fJdT = (2.*n.*(1+n).^2.*(2+n).*r.^(-5-n)*Rd)/(1+2.*n);

        % Build the OUTER quadrature rows (include w and dr for the outer ∫ dr)
        rowJ  = (w(:).'.*dr.* fJT);     % 1-by-xN
        rowJt = (w(:).'.*dr.* fJdT);    % 1-by-xN

        % Convert to rows acting on T and V via KmT
        row_on_TJ = rowJ  * JmT;
        row_on_VJ = rowJt * JmT;

        % Insert into epsilon'' row
        A(idxepd, idxT) = A(idxepd, idxT) +  row_on_TJ;
        A(idxepd, idxV) = A(idxepd, idxV) +  row_on_VJ;

        % second time derivative of T from f_J
        % row that maps Tdd -> integral
        fJdt2 = -n*(n+1)^2/((2*n+1)*R^2 ) * (r(:)).^(-n-2);  % coefficient multiplying K_tt in your f_K term
        row_outer = (w(:).'.*dr.*fJdt2(:).');           % 1-by-xN for ∫ f(r) K_tt(r) dr
        rowTddJ = row_outer * JmT;                    % 1-by-xN so that scalar ≈ rowTddK * Tdd

        TddmodJ = rowTddJ * Li;                          
        A(idxepd, whole_block) = A(idxepd, whole_block) + TddmodJ; 

% ======================================================================
        % Terms from f_K (all from fKintegrandfinal)
% ======================================================================
        % start with termt that modify T
        fKT = n.*(1+n).^2.*r.^(-11+n).*(((1+n).*r.^4.*((2.*r.^3+(-6+n).*R.^3).*Rd.^2+...
            r.^3.*R.*Rdd))./((1+2.*n).*R)+1./(1+2.*n).*((6.*(-1+n.^2).*r.^5.*Rd)./Re+...
            1./(Ca.*rs.^7.*sr.^2).*R/sr.*(-1+sr.^3).*(-((-1+3.*alph).*r.^4.*rs.^2.*(2.*(-2+...
            n).*(-1+n).*r.^6+(6+(11-3.*n).*n).*r.^3.*vs+(-6+n).*(1+n).*vs.^2))+alph.*(2.*(-1+...
            n).*(-16+5.*n).*r.^12+(-10-27.*(-5+n).*n).*r.^9.*vs+(-54+n.*(-177+29.*n)).*r.^6.*vs.^2+...
            3.*(26+(37-5.*n).*n).*r.^3.*vs.^3+3.*(-10+n).*(1+n).*vs.^4))));

        fKdT = 2*(n-1)*n*(n+1).^2.*r.^(n-4).*Rd./(2*n+1);

        % Build the OUTER quadrature rows (include w and dr for the outer ∫ dr)
        rowK  = (w(:).'.*dr.* fKT);     % 1-by-xN
        rowKt = (w(:).'.*dr.* fKdT);    % 1-by-xN

        % Convert to rows acting on T and V via KmT
        row_on_T = rowK  * KmT;
        row_on_V = rowKt * KmT;

        % Insert into epsilon'' row
        A(idxepd, idxT) = A(idxepd, idxT) +  row_on_T;
        A(idxepd, idxV) = A(idxepd, idxV) +  row_on_V;

        % second time derivative of T from f_K
        % row that maps Tdd -> integral
        fKdt2 = n*(n+1)^2/((2*n+1)*R^2 ) * (r(:)).^(n-1);  % coefficient multiplying K_tt in your f_K term
        row_outer = (w(:).'.*dr.*fKdt2(:).');           % 1-by-xN for ∫ f(r) K_tt(r) dr
        rowTddK = row_outer * KmT;                    % 1-by-xN so that scalar ≈ rowTddK * Tdd

        TddmodK = rowTddK * Li;                          
        A(idxepd, whole_block) = A(idxepd, whole_block) + TddmodK; 

    elseif mod == "Pros"

        A(idxepd, idxT(1)) = A(idxepd, idxT(1)) - n*(n+2)*(n+1)./Re*1/R^3;
        A(idxepd, idxT) = A(idxepd, idxT) + n*(n+1)*Rd/R^3*w(:).'.*dr.*dxdr.*(1-(R./r).^3).*(R./r).^n;

        % % include rotational contributions to xi and eta here too 
        Brot = 2*n*(n-1)*(n+1)*Rd/(Re*R^3) - n*(n+1)^2/Ca*1/R^2*sr^(-4);
        Arot = -2*n*(n+2)*(n+1)/(Re*R^2);
        xi = Brot + Arot*Rd/R;
        eta = Arot;  
        A(idxepd, idxep) = A(idxepd, idxep) - xi;
        A(idxepd, idxepd) = A(idxepd, idxepd) - eta ;
    end

end
end
