function [db] = db_pca_mod(X, idx, stop_rule, inputs)
% This function will calculate a metric which could be use for a
% qualitative comparison between different clusterings using VQPCA, as an
% extension of the DB index.
% Inputs:
%   X = Data matrix (enter the scaled data in scaling has been used in
%   VQPCA)
%   idx = clustering labeling vector
%   sq_rec_err = n_obs x k matrix of the reconstruction error for each
%                onbservation in each cluster, element i,j is the rec err of the i-th
%                observation in the j-th cluster
%
%   Output:
%       db = calculated index
%
% DB should be minimized when comparing solutions with different number of
% clusters

% Number of rows and columns
[m,n] = size(X);

% Number of clusters
k = max(idx);

% Check if number of cluster is equal to 1
if k == 1
    disp('Number of clusters should be bigger than 1');
    db = 0.25;
    return
end

% Partition the data according to idx
X_clust = clustering(X, idx);

% Get clusters centroids
C = zeros(k,n);
for i = 1 : k
    C(i,:) = mean(X_clust{i},1);
end

% Get reconstruction error
rec_err_clust = zeros(k,1);
for i = 1 : k

    % Perform PCA
    [sort_eigval, sort_eigvec, ret_eigval, ret_eigvec, n_eig, U_scores, W_scores, gamma, scaled_data, rec_data, X_ave] = ...
        pca_lt(X_clust{i}, 1, 0, stop_rule, inputs);

    % Calculate reconstruction error
    rec_err = sum((X_clust{i}-rec_data).^2);     % Squared reconstruction error for each point

    % Mean squared reconstruction error
    rec_err_clust(i) = mean(rec_err);

end

% Metric initialization and evaluation
db = 0;

for i = 1 : k

    eps_i = rec_err_clust(i);
    db_iter = 0;

    for j = 1 : k

        if j ~= i

            % Average reconstruction error in cluster j
            eps_j = rec_err_clust(j);
            
            % Merge the clusters
            X_ij = [X_clust{i}; X_clust{j}];

            % Perform PCA
            [sort_eigval, sort_eigvec, ret_eigval, ret_eigvec, n_eig, U_scores, W_scores, gamma, scaled_data, rec_data, X_ave] = ...
                pca_lt(X_ij, 1, 0, stop_rule, inputs);

            % Get reconstruction error
            rec_err = sum((X_ij-rec_data).^2);

            % Average reconstruction error of the joined cluster
            eps_ij = mean(rec_err);
            
            % Get the maximum between all the clusters pairs
            db_iter = max(db_iter,(eps_i + eps_j)/eps_ij);

        end
    end

    % Sum for each cluster
    db = db + db_iter;

end

% Average
db = db/k;















