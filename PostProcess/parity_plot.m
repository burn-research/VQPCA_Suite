function [output, r2] = parity_plot(X, rec_data, X_ave_pre, gamma, nz_idx_clust, labels, opt)
% Plot the parity plot: original vs reconstructed data for LPCA. 
% The inputs are all contained in the infos output from local_pca_new.m
% function. Here a summary
% INPUTS
%   rec_data = cell(k,1) array with the unscaled reconstructed data from
%       the local_pca_lt.m function (infos.rec_data)
%
%   X_ave_pre = mxn centering matrix (infos.X_ave_pre)
%
%   gamma = cell(k,1) array containing the pre-scaling factors from
%   local_pca_lt.m (infos.Gamma)
%
%   nz_idx_clust = cell(k,1) containing the indexes of the observations in
%   each clusters
%
%   labels = cell(k,1) containing the names of the variables as string
%
% OUTPUTS
%   output = flag (true or false)
%
%   r2 = R squared coefficient for each variable

% Available options are:
%   opt.Plot = {true, false}
%   opt.Plot = true     ---> The figures are produced
%   opt.Plot = false    ---> Only the r2 is given     
 
% Get data dimension
[n,m] = size(X);        % Observation and features
k = length(rec_data);   % Number of clusters

rec_data_uncentered = unscale_rec(rec_data, nz_idx_clust, gamma, X_ave_pre);

% Initialize R2 vector
r2 = zeros(m,1);

% Parity plot
for j = 1 : m

    if isfield(opt, 'Plot') == false
        opt.Plot = false;
    end
    if opt.Plot == true
        figure;
        scatter(X(:,j), rec_data_uncentered(:,j), 5, 'filled');
        title(labels{j});
        xlabel('Original', 'Interpreter','latex');
        ylabel('Reconstructed', 'Interpreter','latex');
    
        if min(X(:,j)) == 0 && max(X(:,j)) == 0
            xlim([0 1]);
        else
            xlim([min(X(:,j)) max(X(:,j))]);
        end
        
        if min(rec_data_uncentered(:,j)) == 0 && max(rec_data_uncentered(:,j)) == 0
            ylim([0 1])
        else    
            ylim([min(rec_data_uncentered(:,j)) max(rec_data_uncentered(:,j))]);
        end
    end

    % Calculate R2
    r2(j) = 1 - sum ( (X(:,j) - rec_data_uncentered(:,j)).^2 ) / sum ( (X(:,j) - mean(X(:,j))).^2 );

    if opt.Plot == true
        txt = append('$R^2$ = ', num2str(r2(j)));
        text(mean(X(:,j)), mean(X(:,j)), txt, 'fontsize', 18, 'interpreter', 'latex');
        % Line
        x = X(:,j);
        y = x;
        hold on;
        plot(x, y, 'r--', 'LineWidth', 1);
        legend('Reconstructed', 'Original', 'interpreter', 'latex');
        ax = gca; ax.TickLabelInterpreter = 'latex';
        ax.Box = 'on';
    end
end

output = true;
end

