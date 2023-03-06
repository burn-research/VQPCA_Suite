function [cmean, xmean] = conditional_mean(x, y, nbins)
% This function calculates the conditional mean of y given x

% Bin the x variable
h1 = histogram(x, nbins, 'Visible','off');

% Get the bin limits
lim = h1.BinEdges;
xmean = lim(1:end-1)';

cmean = zeros(nbins, 1);
for i = 1 : nbins
    cmean(i) = median(y(x >= lim(i) & x < lim(i+1)));
end



end