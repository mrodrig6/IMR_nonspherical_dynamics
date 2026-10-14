function [wkap] = f_kappa_kernel(s, ds, n, w)
%F_KAPPA_KERNEL Build quadrature weights for the kappa radial kernel.

j1kappa = -(n+1)/(2*n+1).*s.^(-n).*ds;
wkap = w'.*j1kappa;

end
