function ds = f_ds(x, R, L)
%F_DS Compute the radial integration element for the transformed coordinate.

ds = 2*R.*L./(1-x).^2;

end
