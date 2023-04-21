close all;
clear all;

set(0, 'defaulttextfontsize', 20);
set(0, 'defaultaxesfontsize', 20);
set(0, 'defaulttextinterpreter', 'latex');

% Addpath
addpath(genpath('/Users/mattepsavarese/Desktop/Dottorato/Github/LocalPCA_Suite'));

%% Use a more complex dataset
close all;
clear all;

set(0, 'defaulttextfontsize', 20);
set(0, 'defaultaxesfontsize', 20);
set(0, 'defaulttextinterpreter', 'latex');

% Load the data
fold_path = '/Users/matteosavarese/Desktop/Dottorato/Datasets/hydrogen-air-flamelet';

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

%% Create the data matrix by removing useless species
X = state_space_data(:,1:end-3); % Remove N2, Ar, He
T = X(:,1);

% Scatter in the F-T plane
figure;
scatter(f, T, 5, hrr, 'filled');
xlabel('F [-]');
ylabel('T [K]');

cb = colorbar;
cb.Label.Interpreter = 'latex';
cb.Label.String = '$\dot{\omega}_T$';
colormap jet;
cb.TickLabelInterpreter = 'latex';

ax = gca; 
ax.TickLabelInterpreter = 'latex';
fig = gcf;
fig.Units = 'centimeters';
fig.Position = [15 15 16 12];

%% Apply VQPCA, then classify
opt.Center  = 1;
opt.Scaling = 'auto';
opt.Init    = 'uniform1';
[idx, infos] = local_pca_new(X, 3, 1, 0.99, opt);

eigvec = infos.eigenvectors;

X_center = center(X, 1);
X_scaled = scale(X_center, X, 1);
idx_y = classify(X_scaled, X_scaled, idx, eigvec);

figure; subplot(1,2,1);
scatter(f, T, 5, idx, 'filled');
xlabel('F [-]');
ylabel('T [K]');
cb = colorbar;
cb.Label.Interpreter = 'latex';
cb.Label.String = '$\dot{\omega}_T$';
colormap jet;
cb.TickLabelInterpreter = 'latex';
ax = gca; 
ax.TickLabelInterpreter = 'latex';
fig = gcf;
fig.Units = 'centimeters';
fig.Position = [15 15 16 12];

subplot(1,2,2);
scatter(f, T, 5, idx_y, 'filled');
xlabel('F [-]');
ylabel('T [K]');
cb = colorbar;
cb.Label.Interpreter = 'latex';
cb.Label.String = '$\dot{\omega}_T$';
colormap jet;
cb.TickLabelInterpreter = 'latex';
ax = gca; 
ax.TickLabelInterpreter = 'latex';
fig = gcf;
fig.Units = 'centimeters';
fig.Position = [15 15 16 12];








