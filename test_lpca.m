close all;
clear all;

% Addpath
addpath(genpath('/Users/mattepsavarese/Desktop/Dottorato/Github/LocalPCA_Suite'));

X = randn(10000,5);
labels = {'x1', 'x2', 'x3', 'x4', 'x5'};

% Perform local_pca
opt.Scaling = 'range';
opt.Center = 3;

F = linspace(0, 1, length(X))';
Fs = 0.3;

opt.F = F;
opt.FStoich = Fs;
opt.Algorithm = 'VQPCA';

[idx, infos] = local_pca_new(X, 5, 1, 0.9, opt);

% Reconstruct and plot
rec_data_uncentered = unscale_rec(infos.RecData, infos.NzIdxClust, infos.gamma_pre, infos.X_ave_pre);

% Parity plots
opt2.Plot = false;
[output, r2] = parity_plot(X, infos.RecData, infos.X_ave_pre, ...
    infos.gamma_pre, infos.NzIdxClust, labels, opt2);

% Biplot
output = biplot_lpca(infos.eigenvectors, labels);

% Distributions of principal components
output = plot_distributions(infos.UScores);

% Plot weights
opt3.PlotAllClusters = true;
opt3.EigsToPlot = 4;
output = plot_weights(infos.eigenvectors, labels, opt3);