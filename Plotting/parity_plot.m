function [output] = parity_plot(X, X_unscale_rec)
% This function plots the parity plot for each variable

% Check dimensions
[nobs, nvars] = size(X);
if nobs ~= size(X_unscale_rec,1) || nvars ~= size(X_unscale_rec,2)
    error('Dimension of original and reconstructed data must agree');
end

% Plot
for j = 1 : nvars
    figure;
    scatter(X(:,j), X_unscale_rec(:,j), 10, 'filled'); hold on;
    xmin = min(X(:,j)); xmax = max(X(:,j));
    if xmin == xmax
        xmax = xmin + 1e-6;
        mess = append('Equal value of xmin and xmax detected for variable ', num2str(j));
        warning(mess);
    end
    xlim([xmin xmax]);
    ylim([xmin xmax]);
    plot([xmin xmax], [xmin xmax], 'r--', 'LineWidth',1);
    xlabel('True');
    ylabel('Reconstructed');
    ax = gca; ax.Box = 'on';
    fig = gcf;
    fig.Units = 'centimeters';
    fig.Position = [15 15 16 12];

    tit = append('Variable ', num2str(j));
    title(tit);
end

output = true;