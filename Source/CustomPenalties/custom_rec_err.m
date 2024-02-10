function [rec_err] = custom_rec_err(X, C, gamma, eigvec, opt)
% This function can be used to calculate a customized reconstruction error
% for VQPCA. The inputs are from the local_pca_new.m code and they are:
% INPUTS:
%   X     = (np, nv) data matrix (X_scaled from the code)
%   C_mat = (np, nv) centroid matrix for the j-th cluster in the loop
%   gamma = cell(k,1) with the scaling factors for each cluster (normally
%           we do not scale in the loop so it should be ones) 
%   opt = struct array with available options for error
%         e.g. opt.Squared

% Check dimensions of X and C_mat
[rows, columns] = size(X);
if columns ~= size(C, 2)
    error('Dimensions of X and centroid matrix do not agree');
end

% Check the number of clusters
k = length(gamma);

% Initialize reconstruction error
rec_err = zeros(rows, k);    % the element ij represent the reconstruction error 
                            % of the i-th observation in the j-the cluster

%% Various reconstruction error options
% Conventional squared reconstruction error
if isfield(opt, 'Squared')
    for j = 1 : k
        D = diag(gamma{j});
        C_mat = repmat(C(j, :), rows, 1);
        rec_err_os = (X - C_mat - (X - C_mat) * D^-1 * eigvec{j} * eigvec{j}' * D);
        rec_err(:, j) = sum(rec_err_os.^2, 2);
    end

% Reconstruction error is squared root
elseif isfield(opt, 'SquaredRoot')

    for j = 1 : k
        D = diag(gamma{j});
        C_mat = repmat(C(j, :), rows, 1);
        rec_err_os = abs((X - C_mat - (X - C_mat) * D^-1 * eigvec{j} * eigvec{j}' * D));
        rec_err(:, j) = sum(rec_err_os.^0.5, 2);
    end

% Custom power for reconstruction error
elseif isfield(opt, 'CustomPower')
    % Check if a custom power was specified
    if isfield(opt, 'Power')
        pow = opt.Power;
    else
        pow = input('Power was not specified. Specify the power for the reconstruction error: ');
    end

    for j = 1 : k
        D = diag(gamma{j});
        C_mat = repmat(C(j, :), rows, 1);
        rec_err_os = abs((X - C_mat - (X - C_mat) * D^-1 * eigvec{j} * eigvec{j}' * D));
        rec_err(:, j) = sum(rec_err_os.^pow, 2);
    end

% If no options were given use default error
else
    warning(['Custom error option was specified but no options for ...' ...
        'which error or penalty to use were given. Conventional error ...' ...
        'will be used']);

    for j = 1 : k
        D = diag(gamma{j});
        C_mat = repmat(C(j, :), rows, 1);
        rec_err_os = abs((X - C_mat - (X - C_mat) * D^-1 * eigvec{j} * eigvec{j}' * D));
        rec_err(:, j) = sum(rec_err_os.^2, 2);
    end


end

% Check for penalty
if isfield(opt, 'Penalty') && isfield(opt, 'PenaltyNormalized') == false
    if opt.Penalty == true
        penalty = penalty_error(X, C, opt);
        rec_err = rec_err + penalty;
    end

% In this case the penalty is normalized
elseif isfield(opt, 'PenaltyNormalized')
    if opt.PenaltyNormalized == true

        penalty = penalty_error(X, C, opt);
        
        % Check which kind of penalty to apply
        switch opt.NormalizationType
            case 'Max'
       
                % Normalize by the max
                rec_err_norm = zeros(rows, k);
                penalty_norm = zeros(rows, k);
                for j = 1 : k
                    rec_err_norm(:,j) = rec_err(:,j)/max(rec_err(:,j));
                    penalty_norm(:,j) = penalty(:,j)/max(penalty(:,j));
                end

            case 'Mean'
                % Normalize by the max
                rec_err_norm = zeros(rows, k);
                penalty_norm = zeros(rows, k);
                for j = 1 : k
                    rec_err_norm(:,j) = rec_err(:,j)/mean(rec_err(:,j));
                    penalty_norm(:,j) = penalty(:,j)/mean(penalty(:,j));
                end

        end
    end
end

%% Functions with the penalties
function [penalty] = penalty_error(X, C, opt)

% initialize penalty
[rows, ~] = size(X);
k = size(C,1);

penalty = zeros(rows,k);

% Reconstruction error with Euclidean penalty
if isfield(opt, 'EuclideanPenalty')
    % Check for regularization parameters
    if isfield(opt, 'AlphaReg')
        alpha = opt.AlphaReg;
    else
        disp('No regularization parameter was specified, this may distort the results');
        alpha = 1;
    end

    for j = 1 : k

        C_mat = repmat(C(j, :), rows, 1);

        % Euclidean distance
        penalty(:,j) = sum((X - C_mat).^2, 2)*alpha;
    end
end



