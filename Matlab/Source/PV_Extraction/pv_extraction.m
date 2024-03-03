% This is a function for teh selection of the principal variables from
% Principal Components. The model implemented are listed below.

function [ret_data, disc_data, ret_name, disc_name, method] = pv_extraction(data, names, opt)

close all hidden

% Available options to be specified as fields in opt
% opt.Center   = 1 (0 for no centering, 1 for centering)
% opt.Scaling  = {'auto', 'range', 'pareto', 'vast', 'level', 'max'}
% opt.Method   = {'B4', 'B2', 'M2', 'MC', 'PF'}
% opt.Verify   = false (true if check of multivariate nature is preserved)
% opt.StopRule = {'var', 'size', 'stick', 'n_eig', 'large_eig'};
% opt.Inputs   = Input for PCA stop (see pca_lt.m)

%% Method selection
if isfield(opt, 'Method') == false
    method = input('Select the PV method 1)B4 2)B2 3)M_2 4)MC 5)PF \n');
else
    met = opt.Method;
    switch met
        case 'B4'
            method = 1;
        case 'B2'
            method = 2;
        case 'M2'
            method = 3;
        case 'MC'
            method = 4;
        case 'PF'
            method = 5;
        otherwise
            warning('No valid method was specified, please enter the desired method:');
            method = input('Select the PV method 1)B4 2)B2 3)M_2 4)MC 5)PF \n');
    end
end

%% Check centering and scaling
if isfield(opt, 'Center') == false
    cent_crit = 1;
else
    cent_crit = opt.Center;
end

if isfield(opt, 'Scale') == false
    warning('Option for scaling not specified, data will be auto-scaled by default');
    scal_crit = 1;
else
    scal = opt.Scale;
    switch scal
        case 'auto'
            scal_crit = 1;
        case 'range'
            scal_crit = 2;
        case 'pareto'
            scal_crit = 3;
        case 'vast'
            scal_crit = 4;
        case 'level'
            scal_crit = 5;
        case 'max'
            scal_crit = 6;
        otherwise
            warning('No available scaling was selected, data will not be scaled');
    end
end

% PCA stopping rule
if isfield(opt, 'StopRule') == false
    warning('Stopping rule not specified for PCA, variance retained will be used');
    stop_rule = 1;
else
    stop = opt.StopRule;
    switch stop
        case 'var'
            stop_rule = 1;
        case 'size'
            stop_rule = 2;
        case 'stick'
            stop_rule = 3;
        case 'n_eig'
            stop_rule = 4;
        case 'large_eig'
            stop_rule = 5;
        otherwise
            stop_rule = 1;
            warning('No available stopping rule for PCA was selected. Global variance will be used');
    end
end

% PCA inputs
if isfield(opt, 'Inputs') == false
    warning('No inputs for PCA stopping criterion was specified');
    inputs = input('Specify input for PCA: ');
else
    inputs = opt.Inputs;
end


%% ALLOCATION OF VARIABLES

[rows, columns] = size(data);

% Check if columns and label has same length
if columns ~= length(names)
    error('Columns of data matrix differ from number of labels');
end

% Discarded variables
disc_name_num = char(zeros(1, columns));
disc_name = cellstr(disc_name_num);
disc_index = zeros(1, columns);
disc_data = zeros(rows, columns);

% Retained variables
ret_name_num = char(zeros(1, columns));
ret_name = cellstr(ret_name_num);
index = zeros(1, columns);
ret_data = zeros(rows, columns);

%% METHOD B4 FORWARD
%
% In method B4 forward a PCA is performed on the original matrix of k
% variables and n observation, i.e. size(X) = (n, k). The eigenvalues of
% the covariance/correlation matrix are then computed and a criterion is
% chosen to retain a number of them (e.g. Joliffe criterion lambda* = 0.7 *
% ave(lambda)). If p1 components have eigenvalues less than lambda* the
% eigenvectors associated with the remaining k-p1 eigenvalues are
% evaluated starting with the first component. The variable associated with
% the highest eigenvector coefficient is then retained from each of the
% k-p1 variables. A second PCA is performed such that K-p1-p2 variables
% remain. PCA is then repeated until all the components have eigenvalues
% larger than lambda*. 

if method == 1     
    convergence = 0;                       
    red_data = data;
    red_name = names;
    n_disc_name = 0;
    n_disc_data = 0; 
    while convergence == 0
        [PV_rows, PV_columns] = size(red_data);
        redundancy = 0;
        [sort_eigval, sort_eigvec, ret_eigval, ret_eigvec, n_eig] = pca_lt(red_data, cent_crit, scal_crit, stop_rule, inputs);
        if (n_eig == PV_columns)
            convergence = 1;
            continue
        end        
        clear ret_data ret_name index disc_index
        [ret_eigenvectors ret_index] = sort(abs(ret_eigvec), 'descend');
        disp('Indexes of variables in sorted eigenvectors');
        disp(ret_index);
        for i = 1 : n_eig
            if  i == 1
                ret_data(:, i) = red_data(:, ret_index(1, i));
                ret_name(i) = red_name(ret_index(1, i));
                index(i) = ret_index(1, i);
            elseif i > 1
                j = 1;
                for m = 1 : size(index, 2)
                    index_equality = (ret_index(j, i) == index(m));
                    if index_equality == 1
                        redundancy = 1;
                    end
                end

                if redundancy == 0
                    ret_data(:, i) = red_data(:, ret_index(1, i));
                    ret_name(i) = red_name(ret_index(1, i));
                    index(i) = ret_index(1, i);
                elseif redundancy == 1
                    while ((redundancy == 1) && (j <= PV_columns))
                        for m = 1 : size(index, 2)
                            if m == i
                                continue
                            end
                            index_equality(m) = (ret_index(j, i) == index(m));
                        end
                        if (index_equality(:) == 0)
                            redundancy = 0;
                        end
                        j = j+1;
                    end
                    ret_data(:, i) = red_data(:, ret_index(j-1, i));
                    ret_name(i) = red_name(ret_index(j-1, i));
                    index(i) = ret_index(j-1, i);
                end
            end
        end
        p = 1;
        for i = 1 : PV_columns
            if i ~= index(:)
                disc_index(p) = i;
                p = p+1;
            end
        end
        n_disc_var = size(disc_index, 2);
        for i = 1 : n_disc_var
            disc_data(:, i+n_disc_data) = red_data(:, disc_index(i));
            disc_name(i+n_disc_name) = red_name(disc_index(i));
        end
        n_disc_name = size(disc_name, 2);
        n_disc_data = size(disc_data, 2);        
        red_data = ret_data;
        red_name = ret_name;
    end
    ret_names = char(ret_name);
    disc_names = char(disc_name);
    save ret_names.out ret_names -ASCII
    save disc_names.out disc_names -ASCII
    save ret_data.out ret_data -ASCII
    save disc_data.out disc_data -ASCII
    disp('Discarded Variables according to the B4 Forward Method');
    disp(disc_name);
    disp('Retained Variables according to the B4 Forward Method');
    disp(ret_name);
end

%% METHOD B2 BACKWARD

% In method B1 backward a PCA is performed on the original matrix of k
% variables and n observation, i.e. size(X) = (n, k). The eigenvalues of
% the covariance/correlation matrix are then computed and a criterion is
% chosen to retain a number of them (e.g. Joliffe criterion lambda* = 0.7 *
% ave(lambda)). If p1 components have eigenvalues larger than lambda* the
% eigenvectors associated with the remaining k-p1 eigenvalues are evaluated
% starting with the last component. The variable associated with the
% highest eigenvector coefficient is then discarded from each of the k-p1
% variables. A second PCA is performed such that K-p1-p2 variables are
% discarded. PCA is then repeated until all the components have eigenvalues
% larger than lambda*.  

if method == 2     
    convergence = 0;                    % Logical indicator of convergence      
    red_data = data;
    red_name = names;
    n_disc_name = 0;
    n_disc_data = 0;
    while convergence == 0
        [PV_rows, PV_columns] = size(red_data);
        redundancy = 0;
        [sort_eigval, sort_eigvec, ret_eigval, ret_eigvec, n_eig] = pca_lt(red_data, cent_crit, scal_crit, stop_rule, inputs);
        h = (PV_columns - n_eig);
        if (n_eig == PV_columns)
            convergence = 1;
            ret_data = red_data;
            ret_name = red_name;          
            continue
        end        
        clear ret_data ret_name index ret_index disc_index
        [disc_eigenvectors disc_index] = sort(abs(sort_eigvec(:, n_eig + 1 : PV_columns)), 'descend');
        disp('Indexes of variables in sorted eigenvectors');
        disp(disc_index);
        for i = h : -1 : 1
            if  i == h
                disc_data(:, i+n_disc_data) = red_data(:, disc_index(1, i));
                disc_name(i+n_disc_name) = red_name(disc_index(1, i));
                index(i) = disc_index(1, i);
            elseif i < h
                j = 1;                  
                for m = 1 : size(index, 2)
                    index_equality = (disc_index(j, i) == index(m));
                    if index_equality == 1
                        redundancy = 1;
                    end
                end                
                if redundancy == 0
                    disc_data(:, i+n_disc_data) = red_data(:, disc_index(1, i));
                    disc_name(i+n_disc_name) = red_name(disc_index(1, i));
                    index(i) = disc_index(1, i);
                elseif redundancy == 1
                    while ((redundancy == 1) && (j <= PV_columns))
                        for m = h-1 : -1: 1
                            if m == i
                                continue
                            end
                            index_equality(m) = (disc_index(j, i) == index(m));
                        end
                        if (index_equality(:) == 0)
                            redundancy = 0;
                        end
                        j = j+1;
                    end
                    disc_data(:, i+n_disc_data) = red_data(:, disc_index(j-1, i));
                    disc_name(i+n_disc_name) = red_name(disc_index(j-1, i));
                    index(i) = disc_index(j-1, i);
                end
            end
        end
        n_disc_name = size(disc_name, 2);
        n_disc_data = size(disc_data, 2);      
        p = 1;
        for i = 1 : PV_columns
            if i ~= index(:)
                ret_index(p) = i;
                p = p+1;
            end
        end
        n_ret_var = size(ret_index, 2);
        for i = 1 : n_ret_var
            ret_data(:, i) = red_data(:, ret_index(i));
            ret_name(i) = red_name(ret_index(i));
        end  
        red_data = ret_data;
        red_name = ret_name;
    end
    ret_names = char(ret_name);
    disc_names = char(disc_name);
    save ret_names.out ret_names -ASCII
    save disc_names.out disc_names -ASCII
    save ret_data.out ret_data -ASCII
    save disc_data.out disc_data -ASCII
    disp('Discarded Variables according to the B2 Backward Method');
    disp(disc_name);
    disp('Retained Variables according to the B2 Backward Method');
    disp(ret_name);
end

%% METHOD M3 BACKWARD (Krzanowski, 1987):
%
% A SPCA is performed on the original matrix of p variables and n
% observation, i.e. size(x) = (n, p). The eigenvalues of the covariance
% matrix are then computed and a criterion is chosen to retain k of them.
% The (nxk) matrix Y of PCs scores is then evaluated. 
% The goal is to select q (q<p and q>k) variables from the original data
% matrix which preserve most of the data variation. The PCs scores of the
% reduced data will be denoted by �.
% Being k the dimensionality of the data, Y is the �true� configuration
% while � is the corresponding approximation based on q variables. The
% discrepancy between the two configuration is evaluated with a Procrustes
% Analysis. This consists in finding the sum of squared differences between
% corresponding points after they have been matched as well as possible
% under translation, rotation and reflection. Matching under translationis
% ensured by centering both Y and �. Matching under rotation and reflection
% is ensured by considering Y as the fixed configuration and tranforming �.
% The quantity which is to be minimized iun the selection of variables is
% the following sum of squared differences between the configurations:
%
% M2 = (Y'*Y+�'*�-2*S)
%
% where sigma is the matrix of singular values from the SVD of �'Y = USV'
%
% To retain q variables from the original data the following backward
% elimination procedure is employed:
%
% I) Initially, set q=p and, for fixed k, compute the matrix of PCs scores.
% Set Z=Y
% II) Obtain and store the matrix of PCs scores on deleting in turn each
% variable from Z
% III) Compute M2 for each such matrix of scores and identify the variable
% Xu which yields the smallest M2. Let �u denote the corresponding matrix
% of scores
% IV) Delete variable Xu. Set Z= �u and return to stage II with p-1
% variables. Continue the cycle until only q variables are left 

if method == 3     
    red_data = data;
    red_name = names';
    j = 1;
    [r, q] = size(red_data);
    
    % We first perform a PCA analysis on the whole data matrix and we build
    % the matrix of PCs scores by retaining k PCs according to one of the
    % available criterion.
    
    [sort_eigval, sort_eigvec, ret_eigval, ret_eigvec, n_eig, U_scores] = pca_lt(red_data, cent_crit, scal_crit, stop_rule, inputs);
    
    k = n_eig;
    variables = q;
    
    if k == variables;
        error('k MUST be at leat equal to (number_of_variables -1)');
    end
    
    % We have to select the number of variables we want to retain prior to
    % perform the extraction procedure. The number of variables retained
    % MUST be equal or larger than the number of retained PCs.
    
    n_var = input('Select the number of Variables you want to retain \n');
    
    if isempty(n_var)
        n_var = k;
    end
    
    if n_var < k
        error('The number of variables retained MUST be equal or larger than the number of retained PCs');
    end
        
    while q > n_var
        
        [PV_rows, PV_columns] = size(red_data);
        M_2 = zeros(PV_columns);
        M_2_max = 1e+12;
           
        for i = 1 : PV_columns   
            cond_data = [red_data(:, 1:(i-1)) red_data(:, (i+1):PV_columns)];
            cond_name = [red_name(1 : (i-1)) red_name((i+1) : PV_columns)];
            
            if k > size(cond_name, 2)
                continue
            end
            [cond_sort_eigval, cond_sort_eigvec, cond_ret_eigval, cond_ret_eigvec, cond_n_eig, cond_U_scores] = pca_lt(cond_data, 1, 1, 4, k);
            
            % Procrustes analysis to investigate howe the data differ under
            % translation, reflection and rotation
            
            cov_ZY = cond_U_scores' * U_scores;
            [ZY_u, ZY_sv, ZY_v] = svd(cov_ZY, 'econ'); %#ok<NASGU>
            
            % This is the quantity to evaluate to conduct Procrustes
            % Analysis
            
            M_2(i) = trace(U_scores'*U_scores+cond_U_scores'*cond_U_scores-2*ZY_sv);
            
            if M_2(i) < M_2_max
                M_2_max = M_2(i);
                disc_name(j) = red_name(i);
                index(j) = i;
            end

        end
        disc_data(:, j) = red_data(:, index(j));
        ret_data = [red_data(:, 1:index(j)-1) red_data(:, index(j)+1:PV_columns)];
        ret_name = [red_name(1:index(j)-1) red_name(index(j)+1:PV_columns)];        
        red_data = ret_data;
        red_name = ret_name;
        q = size(cond_name, 2);
        j = j+1;
    end
    ret_names = char(ret_name);
    disc_names = char(disc_name);
    save ret_names.out ret_names -ASCII
    save disc_names.out disc_names -ASCII
    save disc_data.out disc_data -ASCII
    save ret_data.out ret_data -ASCII
    disp('Discarded Variables according to the M3 Backward Method');
    disp(disc_name);
    disp('Retained Variables according to the M3 Backward Method');
    disp(ret_name);
end


%% METHOD MC (McCabe, 1984):
%
% The MCs methods by McCabe started from the fact that PCs satisfy a number
% of different optimality criteria. A subset of the original variables that
% optimizes one of these criteria is termed a set of principal variables by
% McCabe (1984):
%
% MC1 = minimize prod(theta_i)*
% MC2 = minimize sum(thetai)*
% MC3 = minimize sum(theta_i^2)*
% MC4 = maximize sum(rho_i^2)*
%
% where theta_i, i = 1, 2,... ,m*, are the eigenvalues of the conditional
% covariance (or correlation) matrix of the m* deleted variables, given the
% values of the m selected variables, and rho j, j= 1, ..., min(m, m*) are
% the canonical correlations between the set of m* deleted variables and
% the set of m selected variables.

if method == 4
    [sort_eigval, sort_eigvec, ret_eigval, ret_eigvec, n_eig, U_scores, W_scores, gamma, scaled_data] = pca_lt(data, cent_crit, scal_crit, stop_rule, inputs);
    q = n_eig;
    ret_index = nchoosek(1 : columns, q);
    n_comb = size(ret_index, 1);
    disp('The number of subsets is equal to ');
    disp(n_comb);
    
    MC_1_min = 1.0e+16;
    MC_2_min = 1.0e+16;
    MC_3_min = 1.0e+16;
    
    opt_crit = input('Optimality criterion: 1) MC1 2) MC2 3) MC3 \n');
    
    for i = 1 : n_comb                
        p = 1;        
        for j = 1 : columns
            if j ~= ret_index(i, :)
                disc_index(p) = j;
                p = p+1;
            end
        end

        sub_data_1 = zeros(rows, q);
        sub_scal_data_1 = zeros(rows, q);
        sub_name_1_num = char(zeros(1, q));
        sub_name_1 = cellstr(sub_name_1_num);
        
        sub_data_2 = zeros(rows, p-1);
        sub_scal_data_2 = zeros(rows, p-1);
        sub_name_2_num = char(zeros(1, p-1));
        sub_name_2 = cellstr(sub_name_2_num);
        
        for j = 1 : q
            sub_data_1(:, j) = data(:,ret_index(i, j));
            sub_scal_data_1(:, j) = scaled_data(:,ret_index(i, j));
            sub_name_1(j) = names(ret_index(i, j));
        end
        
        for j = 1 : (p-1)
            sub_data_2(:, j) = data(:, disc_index(j));
            sub_scal_data_2(:, j) = scaled_data(:, disc_index(j));
            sub_name_2(j) = names(disc_index(j));
        end
       
        cov_data_11 = 1/(rows-1)*(sub_scal_data_1' * sub_scal_data_1);
        cov_data_22 = 1/(rows-1)*(sub_scal_data_2' * sub_scal_data_2);
        cov_data_12 = 1/(rows-1)*(sub_scal_data_1' * sub_scal_data_2);
        cov_data_21 = 1/(rows-1)*(sub_scal_data_2' * sub_scal_data_1);
        
        cov_data_22_1 = cov_data_22 - cov_data_21*(cov_data_11)^-1*cov_data_12;
        eig_cov_data_22_1 = eig(cov_data_22_1);
        
        if opt_crit == 1
            MC_1 = prod(eig_cov_data_22_1);
            if MC_1 < MC_1_min
                MC_1_min = MC_1;
                ret_name = sub_name_1;
                disp('Retained variables names');
                disp(ret_name);
                ret_data = sub_data_1;
                disc_name = sub_name_2;
                disc_data = sub_data_2;
            end
        end
        if opt_crit == 2
            MC_2 = sum(eig_cov_data_22_1);
            if MC_2 < MC_2_min
                MC_2_min = MC_2;
                ret_name = sub_name_1;
                disp('Retained variables names');
                disp(ret_name);
                ret_data = sub_data_1;
                disc_name = sub_name_2;
                disc_data = sub_data_2;
            end
        end
        if opt_crit == 3
            MC_3 = sum(eig_cov_data_22_1.^2);
           if MC_3 < MC_3_min
                MC_3_min = MC_3;
                ret_name = sub_name_1;
                disp('Retained variables names');
                disp(ret_name);
                ret_data = sub_data_1;
                disc_name = sub_name_2;
                disc_data = sub_data_2;
            end
        end
    end
    ret_names = char(ret_name);
    disc_names = char(disc_name);
    save ret_names.out ret_names -ASCII
    save disc_names.out disc_names -ASCII
    save ret_data.out ret_data -ASCII
    save disc_data.out disc_data -ASCII
    disp('Discarded Variables according to the MC Method');
    disp(disc_name);
    disp('Retained Variables according to the MC Method');
    disp(ret_name);
end


%% METHOD PF - Principal Feature Analysis
%
% Let X be a zero mean n-dimensional random feature vector. Let S be the
% covariance matrix of X . Let A be a matrix whose columns are the
% orthonormal eigenvectors of the matrix S:
% S = ALA'
% where L is a diagonal matrix whose diagonal elements are the eigenvalues
% of S. Let Aq be the first q columns of A and let V1,V2 ,...Vn be the rows
% of the matrix Aq. 
% Each vector Vi represents the projection of the i�th feature (variable)
% of the vector X to the lower dimensional space, that is, the q elements
% of Vi correspond to the weights of the i�th feature on each axis of the
% subspace. The key observation is that features that are highly correlated
% or have high mutual information will have similar absolute value weight
% vectors Vi. On the two extreme sides, two independent variables have
% maximally separated weight vectors; while two fully correlated variables
% have identical weight vectors (up to a change of sign). To find the best
% subset we use the structure of the rows Vi to first find the subsets of
% features that are highly correlated and follow to choose one feature from
% each subset. The chosen features represent each group optimally in terms
% of high spread in the lower dimension, reconstruction and insensitivity
% to noise. The algorithm can be summarized in the following five steps:
%
% Step 1 Compute the sample covariance/correlation matrix
%
% Step 2 Compute the Principal components and eigenvalues of the
% covariance/correlation matrix 
%
% Step 3 Choose the subspace dimension q and construct the matrix Aq from
% A.  
%
% Step 4 Cluster the vectors q |V1 |, |V2 |,..., |Vn to p � q clusters
% using K-Means algorithm. The distance measure used for the K-Means
% algorithm is the Euclidean distance. Choosing p greater than q is usually
% necessary if the variability as the PCA is desired
% 
% Step 5 For each cluster, find the corresponding vector Vi which is
% closest to the mean of the cluster. Choose the corresponding feature, xi,
% as a principal feature. This step will yield the choice of p features.
% The reason for choosing the vector nearest to the mean is twofold. This
% feature can be thought of as the central feature of that cluster- the one
% most dominant in it, and which holds the least redundant information of
% features in other clusters. Thus it satisfies both of the properties we
% wanted to achieve- large �spread� in the lower dimensional space, and
% good representation of the original data. 


if method == 5
    [sort_eigval, sort_eigvec, ret_eigval, ret_eigvec, n_eig] = pca_lt(data, cent_crit, scal_crit, stop_rule, inputs);
    q = n_eig;
    
    n_clust = input('Specify the number of clusters \n');
    if isempty(n_clust)
        n_clust = q;
    end
    if n_clust < q
        error('The number of clusters MUST be equal or higher than the number of retained PCs');
    end
    
    [clust_ind, clust_cent] = kmeans(ret_eigvec, n_clust);
    [silh_p, h] = silhouette(ret_eigvec, clust_ind); %#ok<NASGU>
    xlabel('Silhouette Value');
    ylabel('Cluster');
        
    for clust = 1 : n_clust
        dist_min = 1000;
        for i = 1 : columns
            if (clust_ind(i) == clust)
                dist = (ret_eigvec(i, :) - clust_cent(clust, :));
                dist_norm = norm(dist, 2);
                if (dist_norm < dist_min)
                    dist_min = dist;
                    ret_index(clust) = i;
                    ret_name(clust) = names(i);
                    ret_data(:, clust) = data(:, i);
                end
            end
        end
    end
    
    p = 1;

    for j = 1 : columns
        if j ~= ret_index(:)
            disc_index(p) = j;
            p = p+1;
        end
    end

    for i = 1 : (columns - n_clust)
        disc_data(:, i) = data(:, disc_index(i));
        disc_name(i) = names(disc_index(i));
    end
    ret_names = char(ret_name);
    disc_names = char(disc_name);
    save ret_names.out ret_names -ASCII
    save disc_names.out disc_names -ASCII
    save ret_data.out ret_data -ASCII
    save disc_data.out disc_data -ASCII
    disp('Discarded Variables according to the MC Method');
    disp(disc_name);
    disp('Retained Variables according to the MC Method');
    disp(ret_name);
end

mean_ret_data = mean(ret_data, 1);
mean_disc_data = mean(disc_data, 1);
nz_ret_idx = find(mean_ret_data ~=0);
nz_disc_idx = find(mean_disc_data ~= 0);
ret_data = ret_data(:, nz_ret_idx);
disc_data = disc_data(:, nz_disc_idx);
ret_name = ret_name(nz_ret_idx);
disc_name = disc_name(nz_disc_idx);

%% MEASURE OF CLOSENESS BETWEEN RETAINED DATA AND ORIGINAL DATA
%
% Now we can verify if the selected variables preserve the multivariate
% structure of the data


if isfield(opt, 'Verify') == false
    MVDS = false;
else
    MVDS = opt.Verify;
end

if MVDS == true
    [q_sort_eigval, q_sort_eigvec, q_ret_eigval, q_ret_eigvec, q_n_eig, q_U_scores] = pca_lt(ret_data, 1, 1, 1, 1);
    [sort_eigval, sort_eigvec, ret_eigval, ret_eigvec, n_eig, U_scores] = pca_lt(data, 1, 1, 4, n_eig);
    figure;
    subplot(1, 2, 1);
    plot(U_scores(:, 1), U_scores(:, 2), 'b+');
    xlabel('1st Principal Component');
    ylabel('2nd Principal Component');
    title('PCs generated by the original data matrix');
    subplot(1, 2, 2); 
    plot(q_U_scores(:, 1), q_U_scores(:, 2), 'b+')
    xlabel('1st Principal Component');
    ylabel('2nd Principal Component');
    title('PCs generated by the retained variables');
    saveas(gcf, 'Multivariate data structure', 'jpg');
end
        

