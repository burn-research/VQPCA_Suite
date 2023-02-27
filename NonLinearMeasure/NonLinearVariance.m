function [chi, chi_source] = NonLinearVariance(X, nbins, opt)
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


% Get size of X
[np, nv] = size(X);

%% Perform PCA
% Centering
if isfield(opt, 'Center')
    cent_crit = opt.Center;
else
    cent_crit = 1;
    warning('Centering not specified. Data will be centered by default');
end

% Scaling
if isfield(opt, 'Scaling')
    scal_crit = opt.Scaling;
else
    scal_crit = 1;
    warning('Scaling not specified. Data will be scaled with auto scaling by default');
end

% Inputs PCA
if isfield(opt, 'StopRule') && isfield(opt, 'Inputs')
    stop_rule = opt.StopRule;
    inputs = opt.Inputs;
else
    stop_rule = 1;
    inputs = 0.99;
    warning('PCA stop rule and/or inputs not specified correctly. 99% of variance will be retained by default');
end

% Perform PCA
[sort_eigval, sort_eigvec, ret_eigval, ret_eigvec, n_eig, U_scores, W_scores, gamma, scaled_data, rec_data, X_ave] = ...
    pca_lt(X, cent_crit, scal_crit, stop_rule, inputs);

% Scale and center
X_center = center(X, cent_crit);
X_scaled = scale(X_center, X, scal_crit);

%% Calculate conditional mean and conditional variance

u1 = U_scores(:,1);
chi = zeros(nv, 1);

for i = 1 : nv
    [cvar, ~] = conditional_variance(u1, X_scaled(:,i), nbins);
    chi(i) = sum(cvar/var(X_scaled(:,i)))/nbins;
end

%% Check for option for source term
if isfield(opt, 'SourceTerm')
    source = opt.SourceTerm;

    % Check if there are some indexes to remove
    if isfield(opt, 'RemoveIds')
        idr = opt.RemoveIds;
        eigvec = removerows(ret_eigvec, idr);
    else
        eigvec = ret_eigvec;
    end

    % Check shape with source term
    [npt, nvs] = size(source);
    if size(source,2) ~= size(eigvec,1) || length(source) ~= length(X)
        error('Dimension of source term and original space do not match');
    end

    % Project source term
    pc_source = source * eigvec;

    chi_source = zeros(size(pc_source,2), 1);
    for i = 1 : size(pc_source,2)
        [cvar, ~] = conditional_variance(u1, pc_source(:,i), nbins);
        chi_source(i) = sum(cvar/var(pc_source(:,i)))/nbins;
    end

else
    chi_source = [];
end








end

