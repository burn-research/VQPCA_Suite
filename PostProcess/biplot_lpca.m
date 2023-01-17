function [output] = biplot_lpca(eigvec, labels)
% This function will produce a biplot for each cluster, where the weights
% of each variable on the first and second components are represented as
% vectors
n = length(eigvec);

for i = 1 : n
    [m,q] = size(eigvec{i});
    figure;
    eig_clust = eigvec{i};
    if q >1
        b = biplot(eig_clust(:,1:2), 'VarLabels', labels);
        tit = append('Cluster ', num2str(i));
        title(tit, 'interpreter', 'latex');
        % Personalize plot
        for j = 1 : m
            b(j+2*m).FontSize = 8;
        end
    
        ax = gca;
        ax.TickLabelInterpreter = 'latex';
    
        xlabel('PC1', 'Interpreter','latex');
        ylabel('PC2', 'Interpreter','latex');
        ax.Box = 'on';
    else
        
        b = bar(eigvec{i}, 0.8);
        ax = gca;
        ax.TickLabelInterpreter = 'latex';
    
        ax.XTickLabel = labels;
        ylabel('Weight');

end

output = b;



end

