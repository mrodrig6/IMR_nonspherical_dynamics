function [KmT, JmT] = f_J_K_kernel(s, ds, oneW, W, n)
%F_J_K_KERNEL Build quadrature matrices for the J and K radial kernels.

v1 = -s.^(-n).*ds;
KmT = (oneW-W).*v1(:).';

h1 = s.^(n+1).*ds;
JmT = W.*h1(:).';

end
