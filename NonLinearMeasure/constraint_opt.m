function [c, ceq] = constraint_opt(a)
% Constraint for orthogonality
ceq = 1 - a' * a;
c = [];
end