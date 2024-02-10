function [nu] = NonLinearMean(x, y, nbins)
% NonLinearMean.m calculates the average ratio between the gradient of the
% conditional mean of y given x in the bins (nbins) and the global mean of y, as a measure of
% the non linearity between x and y

% Calculate conditional mean of y given x
[cmean_y, xmean_y] = conditional_mean(x, y, nbins);

% Calculate the condtional mean of x given x
[cmean_x, xmean_x] = conditional_mean(x, x, nbins);

cmean_y_der = zeros(length(cmean_y)-1, 1);  % Derivative of conditional mean of y
cmean_x_der = zeros(length(cmean_x)-1, 1);  % Derivative of conditional mean of x

for i = 1 : length(cmean_x)-1

    cmean_y_der(i) = (cmean_y(i+1) - cmean_y(i))/(xmean_y(i+1) - xmean_y(i));
    cmean_x_der(i) = (cmean_x(i+1) - cmean_x(i))/(xmean_x(i+1) - xmean_x(i));

end
    
nu = std( abs(cmean_y_der ./ cmean_x_der));


end

