function [eigvec, n_eigs] = initialize_eigenvectors(X, k, opt)
% This function returns the initialization of the eigenvectors matrix
% eigvec and the eigenvalues gamma. 
% INPUTS:
%   X = data matrix
%   k = number of clusters
%   opt = opt struct dictionary
% 
% OUTPUTS:
%   eigvec = nvar x q eigenvector matrix

% Get infos
[nobs, nvars] = size(X);

% First, get the number of maximum eigenvectors
n_eigs_max = nvars-1;

% Check if the number of initial eigenvectors is specified
if isfield(opt, 'NEigsStart') == false
    fprintf('Number of initial eigenvectors not specified. 2 will be selected as default \n');
    n_eigs_start = 2;
else
    % Check if
    n_eigs_start = int32(opt.NEigsStart);
    if isinteger(n_eigs_start) == false
        warning('Number of starting eigenvectors must be an integer. 2 will be selected as default');
        n_eigs_start = 2;
    end
end

n_eigs = n_eigs_start;

if isfield(opt, 'InitEigs') == false

    % Initialize eigenvectors as a unity matrix
    fprintf('Initialization method for eigenvectors not specified. Identity matrix will be chosen');
    
    eigvec = cell(k, 1);
    for j = 1 : k
        eigvec{j} = eye(nvars, n_eigs_start);
    end

else

    % Check for specified option
    eigs_start = opt.InitEigs;

    % Initialization of centroids from first global PCA
    if strcmp(eigs_start, 'GPCA')

        fprintf('\n Initialization of the eigenvectors from preliminary PCA \n');
        eigvec = cell(k, 1);
        for j = 1 : k
            [~, ~, ~, ret_eigvec, ~, ~, ~, ~] ...
                = pca_lt(X, 1, 0, 4, n_eigs_start);
            eigvec{j} = ret_eigvec;
        end
    end
end



end

