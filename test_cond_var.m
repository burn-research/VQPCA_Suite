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
[cvar, xvar] = conditional_variance(xx, yy, 50);

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
    pca_lt(X, 1, 2, 1, 0.99);

pc_sources = (sources_data(:,1:9)./gamma) * ret_eigvec(1:end,:); 

figure;
scatter3(U_scores(:,1), U_scores(:,2), pc_sources(:,1), 10, hrr, 'filled');
xlabel('$PC_1$');
ylabel('$\dot{\omega}_{PC_1}$');
ax = gca; ax.TickLabelInterpreter = 'latex';
cb = colorbar;
cb.Label.String = '$\dot{\omega}_T$';
cb.Label.Interpreter = 'latex';
cb.TickLabelInterpreter = 'latex';
fig = gcf; fig.Units = 'centimeters';
fig.Position = [15 15 16 12];
colormap('jet');

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
nbins = 0;
opt.Center = 1;
opt.StopRule = 4;
opt.Inputs = 2;

labels = {names{1:9}, '$S_{Z1}$', '$S_{Z2}$', '$S_{Z3}$'};

% Perform PCA with auto scaling
[sort_eigval, sort_eigvec, ret_eigval, ret_eigvec, n_eig, U_scores, W_scores, gamma, scaled_data, rec_data, X_ave] = ...
    pca_lt(X, 1, 4, 4, 3); 

% Project source term
ss_proj = (sources_data(:,1:9)./gamma) * ret_eigvec;

chi_auto = zeros(length(names),3);
for i = 1 : length(labels)
    if i <= 9
        for j = 1 : 3
            chi_auto(i,j) = NonLinearVariance(U_scores(:,j), X(:,i), 25);
        end
    else
        for j = 1 : 3
            chi_auto(i,j) = NonLinearVariance(U_scores(:,j), ss_proj(:,i-9), 25);
        end
    end
end

% Plot
figure;
bar(chi_auto, 0.8);
ax = gca; ax.TickLabelInterpreter = 'latex';
ax.XTickLabel = labels;
legend('$U_1$', '$U_2$', '$U_3$', 'interpreter', 'latex');
fig = gcf; fig.Units = 'centimeters';
fig.Position = [15 15 24 12];
ylabel('$\chi_x$');
ylim([0 2.5]);


%% Test conditional variance on non linear functions

% Line
npt = 1001;
xx = linspace(-1, 1, npt)';
yy = 2*xx - 0.5 + 0.05*randn(npt,1);

% Cubic
yy3 = 2*xx.^3 - xx.^2 + 0.5*xx + 0.05*randn(npt, 1);

% Sine
yys = sin(5*xx) + 0.05*randn(npt,1);

figure;
scatter(xx, yy, 5, 'k', 'filled', 'MarkerFaceAlpha',0.9);
hold on;
scatter(xx, yy3, 5, 'b', 'filled', 'MarkerFaceAlpha',0.9);
scatter(xx, yys, 5, 'r', 'filled', 'MarkerFaceAlpha',0.9);
xlabel('$x_1$');
ylabel('$x_2$');
legend('$x = 2*x - 0.5 + \varepsilon$', '$x = 2*x^3 - x^2 + 0.5x + \varepsilon$', ...
    '$sin(5x) - 0.5 + \varepsilon$', 'interpreter', 'latex', 'Location','southeast');
ax = gca; ax.TickLabelInterpreter = 'latex';
ax.Box = 'on';
fig = gcf; fig.Units = 'centimeters';
fig.Position = [15 15 16 12];

figure;
chi1 = NonLinearVariance(xx, yy, 50);
chi2 = NonLinearVariance(xx, yy3, 50);
chi3 = NonLinearVariance(xx, yys, 50);
close;

figure;
bar([chi1 chi2 chi3], 0.8);
ax = gca; ax.XTickLabel = {'Linear', 'Cubic', 'Sinusoidal'};
ax.TickLabelInterpreter = 'latex';
fig = gcf; fig.Units = 'centimeters';
fig.Position = [15 15 16 12];
ylabel('$\chi_{x2}$');

% Variance of the functions by varying the number of bins
npt = 101;
nb = [10 20 30 40 50 100 200 500 1000];
chi1 = zeros(length(nb), 1);
chi2 = zeros(length(nb), 1);
chi3 = zeros(length(nb), 1);

for i = 1 : length(nb)
    chi1(i) = NonLinearVariance(xx, yy, nb(i));
    chi2(i) = NonLinearVariance(xx, yy3, nb(i));
    chi3(i) = NonLinearVariance(xx, yys, nb(i));

end

figure;
plot(nb, chi1, 'LineWidth',2); hold on;
plot(nb, chi2, 'LineWidth',2);
plot(nb, chi3, 'LineWidth',2);
ax = gca; ax.XScale = 'log';
ax.YScale = 'log';
ax.TickLabelInterpreter = 'latex';
legend('Linear', 'Cubic', 'Sinusoidal', 'interpreter', 'latex');
xlabel('$N_{bins}$');
ylabel('$\chi_{x2}$');
fig = gcf; fig.Units = 'centimeters';
fig.Position = [15 15 16 12];








