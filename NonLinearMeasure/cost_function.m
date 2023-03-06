function [J] = cost_function(a, X, S, alpha)
% project source term
sproj = S * a;
uscores = X * a;

% Reconstruction error
rec_err = mean(sum((uscores*a' - X).^2, 2));

% Check size A
[m, nv] = size(a);

% Cost function
if nv == 1
    J = NonLinearVariance(uscores, sproj, 15) + alpha*rec_err;
else
    J = 0;
    for i = 1 : nv
        Ji = NonLinearVariance(uscores(:,i), sproj(:,i), 15);
        J = J + Ji;
    end
    J = J/nv + alpha * rec_err;
end

end
