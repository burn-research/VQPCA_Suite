function [output] = plot_distributions(U_scores)
% This function plot the distribution of the first two principal component
% scores

k = length(U_scores);

% Matricize the cell
U = [];
for i = 1 : k
    U = [U; U_scores{i}];
end

disp(size(U));

% Histogram
h1 = histogram(U(:,1), 25, 'FaceAlpha',0.5);
hold on;
h2 = histogram(U(:,2), 25, 'FaceAlpha',0.5);

ax = gca;
ax.TickLabelInterpreter = 'latex';

xlabel('PC score', 'interpreter', 'latex');
ylabel('Counts', 'interpreter', 'latex');

legend('PC1', 'PC2', 'Location','northeast', 'interpreter', 'latex');

output = true;

end

