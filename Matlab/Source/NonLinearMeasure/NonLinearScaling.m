function [X_scaled] = NonLinearScaling(X, p)
% This function applies a nonlinear exponential transformation to the
% original data X applying the exponent p:
% X_scaled = sign(X) * abs(X^p)

X_scaled = sign(X) .* abs(X).^p;

end

