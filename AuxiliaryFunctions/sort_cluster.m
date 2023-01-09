function [idx_new] = sort_cluster(X, idx, var_id)
% This function order the cluster in ascending order according to the mean
% value of a selected variable 

[rows, columns] = size(X);
k = max(idx);

if var_id > columns
    error('Please selected a feasible var_id');
end

val = X(:, var_id);

% Partition the selected variable
val_clust = clustering(val, idx);

% Calculate the mean value of the variable in each cluster
var_mean = zeros(k,1);
for i = 1 : k
    var_mean(i) = mean(val_clust{i});
end

% Sort the mean values in ascending order
[var_sort, id] = sort(var_mean, 'descend');
disp(var_mean);
disp(id);
pause;

% Create the new idx vector
idx_new = zeros(rows, 1);
for i = 1 : k
    idx_new(idx == i) = find(id==i);
end





end

