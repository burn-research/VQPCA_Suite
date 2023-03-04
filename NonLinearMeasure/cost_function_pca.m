function [J] = cost_function_pca(gamma, X, S, alpha1, alpha2)
% Check size of gamma
nv = length(gamma);
[~, nv2] = size(X);
if nv ~= nv2
    error('Dimension of gamma is different from X');
end

% Scale the data
X_scaled = X ./ gamma;

% Perform PCA
[sort_eigval, sort_eigvec, ret_eigval, ret_eigvec, n_eig, U_scores, W_scores, gamma_pca, scaled_data, rec_data, X_ave] = ...
    pca_lt(X_scaled, 0, 0, 4, 2); 

% project source term
sproj = (S./gamma) * ret_eigvec;

% Reconstruction error
rec_err = mean(sum((X_scaled - rec_data).^2), 2);

% Conditional variance
chi = NonLinearVariance(U_scores(:,1), sproj(:,1), 25);

% Cost function
J = alpha1 * chi + alpha2 * norm(gamma);


end