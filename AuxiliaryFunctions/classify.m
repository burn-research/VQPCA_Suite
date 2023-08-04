<<<<<<< HEAD
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






=======
function [idx_c] = classify(X, Y, idx, eigs)
% idx_c = classify(X, Y, idx)
% This function classify new observations of a matrix Y following the
% obtained VQPCA solution for a given matrix X, where eigs is the set of
% eigenvectors in each cluster
% INPUTS:
%   X = scaled and centered training matrix 
%   Y = scaled and centered test matrix
%   idx = clustering vector (same length as X)
%   eigs = eigenvectors in each cluster (cell x k)
%   
% OUTPUTS
% idx_c = new clustering vector for the matrix Y

% Dimension check
[np, nv] = size(X);
if np ~= length(idx)
    error('X and idx should have same length');
end

[npy, nvy] = size(Y);

if nv ~= size(Y, 2)
    error('X and Y must have same number of variables');
end

k = max(idx);
if k ~= length(eigs)
    error('Length of eigs must be equal to k');
end

% Partition the data and calculate centroids
Xc = clustering(X, idx);
C = zeros(k, nv);
for i = 1 : k
    C(i,:) = mean(Xc{i});
end

% For each observation calculate the squared reconstruction error
sq_rec_err = zeros(npy, k);
for j = 1 : k
    C_mat = repmat(C(j,:), npy, 1);
    rec_err_os = (Y - C_mat - (Y - C_mat)  * eigs{j} * eigs{j}');
    sq_rec_err(:, j) = sum(rec_err_os.^2, 2);
end

% Classical reconstruction error
[~, idx_c] = min(sq_rec_err, [], 2);

end
>>>>>>> tmp

