% Condition.m - This function perform data conditioning
%
% function [bin_data, idx_clust] = condition(data, z, n_bins, opt) 
%
% INPUTS
% 
% data            = Matrix of state variables
% z               = Conditioning variable vector
% n_bins          = Number of bins to condition the data into
% opt             = Selected options (see below)
% OPTIONAL:
% opt.MinZ          = Minimum value of the coinditioning variable (if not
%                   selected the minimum absolute value of z will be used)
%
% opt.MaxZ          = Maximum value of the coinditioning variable (if not
%                   selected the maximum absolute value of z will be used)
%
% opt.Fs            = If conditioning on mixture fraction, the value of the
%                   stoichiometric mixture fraction can be provided. If
%                   not, a uniform partition (into uniform bins) is
%                   performed   
% OUTPUTS
%
% bin_data        = Cell matrix od conditioned data. Each cell stores the
%                   data corresponding to a single bin
% idx_clust       = Cell vector storing the indexes of the original data
%                   points in the bins. 

function [bin_data, idx_clust] = fpca_new(data, z, n_bins, opt)

% Check the inputs
if size(z, 2) > 1
    error('The conditioning variable MUST be a vector');
end

if size(data, 1) ~= size(z, 1)
    error('Inputs do not have the same length')
end

%% Select main options

% Check for user selected zmin
if isfield(opt, 'MinZ')
    min_z = opt.MinZ;
else
    min_z = min(z);
end

% Check for user selected zmax
if isfield(opt, 'MaxZ')
    max_z = opt.MaxZ;
else
    max_z = max(z);
end

% Check for user selected sampling method
if isfield(opt, 'Sampling') == false
    cond_type = 'uniform';
    warning('opt.Sampling field not specified. Uniform sampling will be applied by default');
else
    cond_type = opt.Sampling;

    % Check existence
    if strcmp(cond_type, 'uniform') == false && strcmp(cond_type, 'non_uniform') == false
        warning(['Sampling method specified not existing. Select uniform or non_uniform. \n' ...
            'Uniform will be selected as default']);
        cond_type = 'uniform';
    end
    
    % Check if stoichiomnetric mixture fraction is specified
    if strcmp(cond_type, 'non_uniform') == true
        if isfield(opt, 'Fs') == false
            error('Stoichiometric mixture fraction (or non-uniform reference value) not specified');
        else
            z_stoich = opt.Fs;
        end
    end
end

% Number of intervals

n = n_bins + 1;

% Partition of the space

if n_bins == 1
    ints = linspace(min_z, max_z, n);
elseif strcmp(cond_type, 'uniform')
    ints = linspace(min_z, max_z, n);
elseif strcmp(cond_type, 'non_uniform')
    ints_1 = linspace(min_z, z_stoich, ceil(n/2));
    ints_2 = linspace(z_stoich, max_z, ceil((n+1)/2));
    ints = [ints_1(1:ceil(n/2-1)) ints_2];
end

% Bin data matrices initialization

bin_data = cell(n_bins, 1);
idx_clust = cell(n_bins, 1);

%  Partition

for bin = 1 : n_bins  
    idx_clust{bin} = find((z >= ints(bin)) & (z <= ints(bin+1)));
    bin_data{bin} = data(idx_clust{bin}, :);
end