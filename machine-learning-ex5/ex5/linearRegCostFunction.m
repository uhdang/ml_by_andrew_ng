function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% printf("theta: %d %d\n", size(theta));
% printf("X: %d %d\n", size(X));
% printf("y: %d %d\n", size(y));

% ======= Cost Function ========
% theta_reg - replace the first element to 0 OR just slice the first element out
theta_reg = theta(2:end);

h_x = X * theta; % No sigmoid

reg = (lambda / (2 * m)) * (sum(theta_reg .^ 2));
cost_reg = (1 / (2 * m)) * sum((h_x - y).^2) + reg;

J = cost_reg;

% ======= Gradient ========
theta_grad_reg = [zeros(1, size(theta)(2)) ; theta(2:end)];
grad = ((1 / m) * (h_x - y)' * X)' + ((lambda / m) .* theta_grad_reg);

% =========================================================================

grad = grad(:);

end
