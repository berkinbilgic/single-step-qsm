function [ y ] = CFL2_Edge( X, D2, mu, fdx, fdy, fdz, cfdx, cfdy, cfdz, magn_weight )
%APPLY_FORWARD Summary of this function goes here
%   Detailed explanation goes here

x = reshape(X, size(D2));

y = D2 .* x + mu .* (cfdx.*fftn(magn_weight(:,:,:,1) .* ifftn(fdx.*x)) + ...
    cfdy.*fftn(magn_weight(:,:,:,2) .* ifftn(fdy.*x)) + cfdz.*fftn(magn_weight(:,:,:,3) .* ifftn(fdz.*x)));

y = y(:);

end