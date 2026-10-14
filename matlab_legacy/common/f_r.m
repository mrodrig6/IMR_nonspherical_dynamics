function r = f_r(x, R, L)
%compute_map_x_to_r is purely to calculate the map to r given x
% x and r are the same datatype, either scalar or vector depending on input

r = R.*((1+x)./(1-x).*L+1);

end