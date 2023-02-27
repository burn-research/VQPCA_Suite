close all;
clear all;

set(0, 'defaulttextfontsize', 20);
set(0, 'defaultaxesfontsize', 20);
set(0, 'defaulttextinterpreter', 'latex');

% Addpath
addpath(genpath('/Users/mattepsavarese/Desktop/Dottorato/Github/LocalPCA_Suite'));

%% Create a non-linear curve with some noise
xx = linspace(-1, 1, 2001)';
yy = 1 - 0.5*xx + 0.3*xx.^2 + 0.5*randn(2001,1).*xx;

figure;
scatter(xx, yy, 10, 'k', 'filled', 'MarkerFaceAlpha',0.5);
hold on;

% Calculate conditional mean and conditional variance
[cmean, xmean] = conditional_mean(xx, yy);
[cvar, xvar] = conditional_variance(xx, yy);

plot(xmean, cmean, 'r-', 'LineWidth',2);

figure;
plot(xvar, cvar, 'b-', 'LineWidth',2);

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
cb.Label.String = 'Hrr $[W/m^3]$';
cb.Label.Interpreter = 'latex';
colormap jet;
cb.TickLabelInterpreter = 'latex';

ax = gca; 
ax.TickLabelInterpreter = 'latex';
fig = gcf;
fig.Units = 'centimeters';
fig.Position = [15 15 16 12];

%% Project source term of the PC
% Apply PCA

% Perform PCA
[sort_eigval, sort_eigvec, ret_eigval, ret_eigvec, n_eig, U_scores, W_scores, gamma, scaled_data, rec_data, X_ave] = ...
    pca_lt(X, 1, 3, 1, 0.99);

pc_sources = sources_data(:,1:8) * ret_eigvec(2:end,:); 

figure;
scatter(U_scores(:,1), pc_sources(:,1), 10, hrr, 'filled');
xlabel('$PC_1$');
ylabel('$\dot{\omega}_{PC_1}$');
ax = gca; ax.TickLabelInterpreter = 'latex';
cb = colorbar;
cb.Label.String = '$\dot{\omega}_T$';
cb.Label.Interpreter = 'latex';
cb.TickLabelInterpreter = 'latex';
fig = gcf; fig.Units = 'centimeters';
fig.Position = [15 15 16 12];

figure;
scatter(hrr, pc_sources(:,1), 10, 'k', 'filled');
ylabel('$PC_1$');
xlabel('$\dot{\omega}_T$');
ax = gca; ax.TickLabelInterpreter = 'latex';
fig = gcf; fig.Units = 'centimeters';
fig.Position = [15 15 16 12];


%% Test different scaling and the non-linear measure
% Calculate conditional variance
clear opt;
nbins = 20;
opt.Center = 1;
opt.StopRule = 4;
opt.Inputs = 2;

opt.Scaling = 1;
opt.SourceTerm = sources_data(:,1:8);
opt.RemoveIds = 1;
[chi1, chi_source1] = NonLinearVariance(X, nbins, opt);

opt.Scaling = 2;
[chi2, chi_source2] = NonLinearVariance(X, nbins, opt);

opt.Scaling = 3;
[chi3, chi_source3] = NonLinearVariance(X, nbins, opt);

chi1 = [chi1; chi_source1];
chi2 = [chi2; chi_source2];
chi3 = [chi3; chi_source3];

% Calculate the ones for SZ1 and SZ2
figure;
xbar = [chi1 chi2 chi3];
b = bar(xbar);

cmap = brewermap(3, 'Greens');
b(1).FaceColor = cmap(1,:);
b(2).FaceColor = cmap(2,:);
b(3).FaceColor = cmap(3,:);

ax = gca; ax.TickLabelInterpreter = 'latex';
ax.XTickLabel = {names{1:9}, 'SZ1', 'SZ2'};
legend('auto', 'range', 'pareto', 'interpreter', 'latex', 'Location','northwest');

ylabel('$\chi_\phi^{(i)}$');

fig = gcf; fig.Units = 'centimeters';
fig.Position = [15 15 20 14];









