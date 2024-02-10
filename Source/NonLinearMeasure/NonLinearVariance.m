function [chi] = NonLinearVariance(x, y, nbins)
% NonLinearVariance implements the metric of Isaac et al. in evaluating the
% non linearity of a certain variable. 
% INPUTS:
%   X = np x nv data matrix
%   nbins = number of bins
%   opt = option struct field
%       opt.Scaling = (int) scaling criterion
%       opt.Center  = 1 or 0 
%       opt.SourceTerm = source term in the original state-space
%       opt.RemoveIds  = list of rows to remove from eigenvectors if shape
%                        do not match
%
% OUTPUTS:
%   chi = nv normalized variance

%% Calculate conditional mean and conditional variance

[cvar, ~] = conditional_variance(x, y, nbins);
chi = sum(cvar/var(y))/nbins;



end


