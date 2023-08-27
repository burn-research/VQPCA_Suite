close all;
clear all;

% Addpath
addpath(genpath('/Users/mattepsavarese/Desktop/Dottorato/Github/LocalPCA_Suite'));

%% Load data
data_ss = importdata('TestData/hydrogen-air-flamelet/STEADY-clustered-flamelet-H2-state-space.csv');
data_f  = importdata('TestData/hydrogen-air-flamelet/STEADY-clustered-flamelet-H2-mixture-fraction.csv');

fs = 0.19;

%% FPCA
% Perform local_pca
opt.Scaling = 'pareto';
opt.Center = 1;
opt.Algorithm = 'FPCA';
opt.Init = 'uniform';
opt.F = data_f;

[idx, infos] = local_pca_new(data_ss, 10, 4, 2, opt);

[bin_data, idx_clust, idx_fpca] = fpca_new(data_ss, data_f, 10, opt);

scatter(data_f, data_ss(:,1), 10, idx_fpca, 'filled');