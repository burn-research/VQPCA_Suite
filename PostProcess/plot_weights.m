function [output] = plot_weights(eigenvectors, labels, opt)
% This function plot the weights of the variables associated with the
% eigenvectors
% INPUTS
%   eigenvectors = n x q matrix of retained eigenvectors
%   labels = cell(n,1) array with the names of the variables

% Number of variables
[n,~] = size(eigenvectors{1});

if n ~= length(labels)
    error('Length of labels incoherent with size of eigenvectors, %d vs %d', n, length(labels));
end

% Number of clusters
k = length(eigenvectors);

if isfield(opt, 'PlotAllClusters') == true

    for i = 1 : k
        eigs  = eigenvectors{i};
        [n,q] = size(eigs);

        if isfield(opt, 'PlotAllEigenvectors') == true
            figure;
            bar(eigs);
            leg = cell(q,1);
            for j = 1 : q
                leg{j} = append('PC', num2str(j));
            end


        elseif isfield(opt, 'EigsToPlot') == true
            nq = opt.EigsToPlot;
            figure;
            b = bar(eigs(:,1:nq), 'FaceColor','flat');
            leg = cell(nq,1);
            for j = 1 : nq
                leg{j} = append('PC', num2str(j));
            end

             
            cmap = colormap('copper');
            for j = 1:nq
                b(j).FaceColor = cmap(j*length(cmap)/nq,:);
            end

        end

        ax = gca;
        ax.TickLabelInterpreter = 'latex';
        ax.XTickLabels = labels;
        ax.TickLabelInterpreter = 'latex';
        ylabel('Weights', 'interpreter', 'latex');
        legend(leg, 'Location','bestoutside', 'interpreter', 'latex');

        fig = gcf;
        fig.Units='centimeters';
        fig.Position = [15 15 16 12];
    end
end

output = gcf;




end

