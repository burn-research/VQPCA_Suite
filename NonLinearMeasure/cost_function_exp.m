function [J] = cost_function_exp(p, X, S, alpha1, alpha2)

% Scale the data
X_scaled = NonLinearScaling(X, p);

% Perform PCA
[sort_eigval, sort_eigvec, ret_eigval, ret_eigvec, n_eig, U_scores, W_scores, gamma_pca, scaled_data, rec_data, X_ave] = ...
    pca_lt(X_scaled, 0, 0, 4, 2); 

% project source term
sproj = (NonLinearScaling(S, p)) * ret_eigvec;

% Reconstruction error
rec_err = mean(sum((X_scaled - rec_data).^2), 2);

cost = 'variance';
switch cost
    case 'variance'
        % Conditional variance
        chi = NonLinearVariance(U_scores(:,1), sproj(:,1), 25);
        
        % Cost function
        J = alpha1 * chi;

    case 'mean'

        % Conditional variance
        chi = NonLinearMean(U_scores(:,1), sproj(:,1), 25);
        
        % Cost function
        J = alpha1 * chi;

end
