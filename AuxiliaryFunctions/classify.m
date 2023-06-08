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

