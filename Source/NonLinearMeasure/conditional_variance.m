function [cvar, xmean] = conditional_variance(x, y, nbins)
% This function calculates the conditional variance of y given x and the number
% of bins nbins. cvar is the conditional variance while xvar is the
% variable sample space

% Bin the x variable
h1 = histogram(x, nbins, 'Visible','off');

% Get the bin limits
lim = h1.BinEdges;
xmean = lim(1:end-1)';

cvar = zeros(nbins, 1);
for i = 1 : nbins
    cvar(i) = var(y(x >= lim(i) & x < lim(i+1)));

    % Check if is NaN
    if isnan(cvar(i))

        % Check if bin is empty
        np = length(y(x >= lim(i) & x < lim(i+1)));
        if np == 0
            warning('Empty bin detected');
        end
        
        % Get how many counters you can iterate
        counter = nbins-1-i;
        count = 1;
        while isnan(cvar(i)) || counter > 0
            cvar(i) = var(y(x >= lim(i) & x < lim(count+1)));
            count = count+1;
            counter = counter - 1;
            disp('Iterating to find cvar...');
        end

    end
end
