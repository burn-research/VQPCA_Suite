function [rec_data_uncentered] = unscale_rec(rec_data, nz_idx_clust, gamma, X_ave_pre)
% This function will unscale and uncenter the reconstructed data from LPCA

[n,m] = size(X_ave_pre);    % Size of the original data
k = length(rec_data);       % Number of clusters

rec_data_scaled = zeros(n,m);
rec_data_unscaled = zeros(n,m);

for j = 1 : length(rec_data)
    X_clust = rec_data{j};
    idx_clust = nz_idx_clust{j};
    rec_data_scaled(idx_clust,:) = X_clust;
    rec_data_unscaled(idx_clust,:) = unscale(X_clust, gamma);
end


rec_data_uncentered = uncenter(rec_data_unscaled, X_ave_pre);

end

