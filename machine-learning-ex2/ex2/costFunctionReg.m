function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% z = X * theta;
% h_theta_x = sigmoid(z);

% cost_wo_reg = (-transpose(y) * log(h_theta_x) - transpose(1 - y) * log(1 - h_theta_x)) / m;
theta_wo_first_row = theta;
theta_wo_first_row(1,:) = [0];

cost_reg_factor = (lambda / (2 * m)) * (transpose(theta_wo_first_row) * theta_wo_first_row);

grad_reg_factor = (lambda / m) * theta_wo_first_row;

[J, grad] = costFunction(theta, X, y);
J = J + cost_reg_factor;

grad = grad + grad_reg_factor;





% =============================================================

end
