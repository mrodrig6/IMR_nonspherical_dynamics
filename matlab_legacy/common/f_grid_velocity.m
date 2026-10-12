function a = f_grid_velocity(x, R, Rd, L)
%F_GRID_VELOCITY Compute the transformed-grid velocity for the moving map.

a = Rd/R.*(1-x).^2./(2*L).*(1+L.*(1+x)./(1-x));

end
