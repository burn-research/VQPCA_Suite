function [C] = initialize_centroids(X, k, opt)
% This function returns k centroids given the matrix X and the options
% specified in opt.
% Available options are:
%   'random':       pick k random samples
%
%   'uniform1':     pick k samples sampled uniformly
%
%   'PCA'           perform PCA and sample uniformly within the U_scores
%
%   'best_DB'        perform N initial iterations and pick the solution with
%                   best DB
%
%   'uniform2':     Initialize idx(i) uniformly
%
%   'uniform3':     Initialize idx(i) uniformly but first sort the first
%                   column of X

% Default value for n_eigs
n_eigs = 2;

[rows, columns] = size(X);

if isfield(opt, 'Init') == false
    init = 'uniform1';
else
    init = opt.Init;
end

% Uniform1
if strcmp(init, 'uniform1')
    C_int = linspace(1, rows, k+2);
    C = X(round(C_int(2:k+1)), :);

% Random
elseif strcmp(init, 'random')
    C_int = randsample(rows, k);
    C = X(C_int, :);

% From preliminary PCA
elseif strcmp(init, 'PCA')
    
    % Random initialization from PC scores
    [~, ~, ~, ~, n_eig, U_scores, ~, ~, ~, ~] = ...
        pca_lt(X, pre_cent_crit, pre_scal_crit, 1, 0.95);
    
    % Range of the U_score matrix
    inter = linspace(min(U_scores(:,1)), max(U_scores(:,1)), k+1);
    
    % Pick all the values in between the intervals and randomly
    % select an index from them
    C_int = zeros(k,1);
    for i = 1 : length(inter) - 1
        samples = find(U_scores(:,1) > inter(i) & U_scores(:,1) < inter(i+1));
        ind = randsample(length(samples), 1);
        C_int(i) = ind;
    end
    
    % Centroids initialization
    C = X(C_int, :);

% Initialize with best random of 10 initial iterations
elseif strcmp(init, 'best_DB')
    it_init = 20;
    fprintf('Initializing from the best of initial %d iterations /n', it_init);
    rec_err_init = zeros(it_init, 1);
    C_init = cell(it_init, 1);
    db_init = zeros(it_init, 1);
    for i = 1 : it_init
        C_int = randsample(rows, k);
        C = X(C_int, :);
        C_init{i} = C;
        
        % Init eigenvectors
        eigvec = cell(k, 1);
        gamma = cell(k, 1);
        sq_rec_err = zeros(rows, k);
        for j = 1 : k
            eigvec{j} = eye(columns, n_eigs);
            gamma{j} = ones(1, columns);
        
            % Evaluate reconstruction error at starting point
            D = diag(gamma{j});
            C_mat = repmat(C(j, :), rows, 1);
            rec_err_os = (X - C_mat - (X - C_mat) * D^-1 * eigvec{j} * eigvec{j}' * D);
            sq_rec_err(:, j) = sum(rec_err_os.^2, 2);        
        end
        
        
        rec_err_init(i) = mean(min(sq_rec_err, [], 2));
        [~, idx_init] = min(sq_rec_err, [], 2);
        
        db_init(i) = db_pca_mod(X, idx_init, 1, 0.99);
        
    end
    
    % Pick minimum from all the iterations
    [~, id_min] = min(rec_err_init);
    C = C_init{id_min};
    
% Uniform 2
elseif strcmp(init, 'uniform2')
    
    spacing = ceil(rows/k);
    idx = zeros(rows,1);
    
    for i = 1 : rows
        idx(i) = ceil(i/spacing);
    end
    
    data_part = clustering(X, idx);
    
    C = zeros(k,columns);
    for i = 1 : length(data_part)
        C(i,:) = mean(data_part{i}, 1);
    end
    
    opt_3 = 'uniform2';

% Uniform 3
elseif strcmp(init, 'uniform3')

    [~,index] = sort(X(:,1), 'ascend');
    X = X(index,:);
    C_int = linspace(1, rows, k+2);
    C = X(round(C_int(2:k+1)), :);
    opt_3 = 'uniform3';

else
    warning('Not avaliable initialization method was selected, uniform1 will be selected by default');
    C_int = linspace(1, rows, k+2);
    C = X(round(C_int(2:k+1)), :);
end


end

