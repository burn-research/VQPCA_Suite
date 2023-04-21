function [idx_y, rec_err_y] = classify(X, Y, idx, eigvec)
%
% idx_y = classify(X, Y, idx)
%
% This function, given the training matrix X and the trained clustering
% vector idx, retrieve the new labelling vector idx_y for the matrix Y
% using local PCA and minizing the reconstruction error
% opt is the option struct and must be coherent with the one used for VQPCA

% Check on dimensions
[m1, n1] = size(X);
[m2, n2] = size(Y);

if n1 ~= n2
    error('Dimension of X and Y must agree');

elseif m1 ~= length(idx)
    error('Length of idx and X must agree');
end

% Get number of clusters
k = max(idx);
if k ~= length(eigvec)
    error('Number of cluters k and length of eigs must agree');
end

% Get centroids of X
X_clust = clustering(X, idx);
C = zeros(k, n1);
for i = 1 : k
    C(i,:) = mean(X_clust{i});
end

% Get reconstruction error for Y
sq_rec_err = zeros(m2, k);
for j = 1 : k
    C_mat = repmat(C(j,:), m2, 1);
    rec_err_os = (Y - C_mat - (Y - C_mat) * eigvec{j} * eigvec{j}');
    sq_rec_err(:, j) = sum(rec_err_os.^2, 2);
end

% Get the new reconstruction error
[rec_err_min, idx_y] = min(sq_rec_err, [], 2);

% Get the mean reconstruction error for Y
rec_err_y = mean(rec_err_min);







