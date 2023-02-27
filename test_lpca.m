close all;
clear all;

% Addpath
addpath(genpath('/Users/mattepsavarese/Desktop/Dottorato/Github/LocalPCA_Suite'));

% Import data from CFD
data_solution = importdata('/Users/matteosavarese/Desktop/Dottorato/Data_Furnace/Data_CFD/2D/25mm/50CH4_50H2_phi08_kee/data_solution');
val = data_solution.data;
X = val(:,4:end);
labels = data_solution.textdata;

% Perform local_pca
opt.Scaling = 'range';
opt.Center = 1;

opt.Algorithm = 'VQPCA';

% Use customization error?
opt.CustomError = false;

% Use error penalty?
opt.EuclideanPenalty = true;
opt.Penalty = true;
opt.AlphaReg = 0.001;
opt.PenaltyNormalized = true;
opt.NormalizationType = 'Max';

% Select custom power
opt.CustomPower = true;
opt.Power = 3;
opt.Init = 'best_DB';

k = 7;
[idx, infos] = local_pca_new(X, k, 1, 0.99, opt);

% Plot
figure;
scatter(val(:,3), val(:,2), 10, idx, 'filled');
cmname = append('parula(', num2str(k), ')');
colormap(cmname);
cb = colorbar;
fig = gcf; fig.Units = 'centimeters';
fig.Position = [15 15 14 12];

! rm -r VQPCA_*

% Use k-means for a comparison
X_center = center(X, 1);
X_scaled = scale(X_center, X, 2);
idx_kmeans = kmeans(X_scaled, k, 'start', 'plus', 'MaxIter',1000);
figure;
scatter(val(:,3), val(:,2), 10, idx_kmeans, 'filled');
cmname = append('parula(', num2str(k), ')');
colormap(cmname);
cb = colorbar;
fig = gcf; fig.Units = 'centimeters';
fig.Position = [15 15 14 12];
