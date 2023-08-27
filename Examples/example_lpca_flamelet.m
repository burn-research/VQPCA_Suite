close all;
clear all;

% Set default appearence
set(0, 'defaulttextfontsize', 20);
set(0, 'defaultaxesfontsize', 20);

%% Import data

% Load the data
fold_path = 'TestData/hydrogen-air-flamelet';

% State space
state_space_name = append(fold_path, '/STEADY-clustered-flamelet-H2-state-space.csv');
state_space_data = importdata(state_space_name);

% Sources
sources_name = append(fold_path, '/STEADY-clustered-flamelet-H2-state-space-sources.csv');
sources_data = importdata(sources_name);

% Mixture fraction
mixture_fraction_name = append(fold_path, '/STEADY-clustered-flamelet-H2-mixture-fraction.csv');
f = importdata(mixture_fraction_name);

% Heat release rate
hrr_name = append(fold_path, '/STEADY-clustered-flamelet-H2-heat-release-rate.csv');
hrr = importdata(hrr_name);

% Dissipation rate
diss_rate_name = append(fold_path, '/STEADY-clustered-flamelet-H2-dissipation-rates.csv');
diss_rate = importdata(diss_rate_name);

% Names
names = append(fold_path, '/STEADY-clustered-flamelet-H2-state-space-names.csv');
names = importdata(names);

%% Plot data

figure;
scatter(f, state_space_data(:,1), 5, sources_data(:,1), 'filled');
% Set axis labels
xlabel('Z [-]'); ylabel('T [K]');
% Colormap and colorbar
colormap jet;
cb = colorbar;
cb.Label.String = 'heat release rate [W/m3]';
% Figure size
fig = gcf; fig.Units = 'centimeters';
fig.Position = [15 15 16 12];

%% Perform LPCA
opt.Centering = 1;
opt.Scaling = 'auto';
opt.Init = 'uniform1';
opt.Algorithm = 'VQPCA';

% Select to retain the 99% of the variance in each cluster
stop_rule = 1;
var = 0.99;

% Select number of clusters
k = 4;

% Select data
X = state_space_data;

[idx, infos] = local_pca_new(X, k, 1, 0.99, opt);

%% Scatter plot of results
figure;
scatter(f, state_space_data(:,1), 5, idx, 'filled');
% Set axis labels
xlabel('Z [-]'); ylabel('T [K]');
% Colormap and colorbar
cmap = append('parula(', num2str(k), ')');
colormap(cmap);
cb = colorbar;
cb.Label.String = 'Cluster';
cb.Ticks = [1:1:k];
% Figure size
fig = gcf; fig.Units = 'centimeters';
fig.Position = [15 15 16 12];

%% Parity plot

rec_data = infos.RecData;       
nz_idx_clust = infos.NzIdxClust;
gamma_pre = infos.gamma_pre;
X_ave_pre = infos.X_ave_pre;

% Reconstruct the data
[rec_data_uncentered] = unscale_rec(rec_data, nz_idx_clust, gamma_pre, X_ave_pre);

% Parity plot
output = parity_plot(X, rec_data_uncentered);
