function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% theta's dim => 3 x 1
% X's dim => 20 x 3
% y's dim => 20 x 1

z = X * theta; 								% 20 x 1
h_theta_x = sigmoid(z); 					% 20 x 1
left =  -transpose(y) * log(h_theta_x);
right = transpose(1 - y) * log(1 - h_theta_x);
J = (left - right) / m;

grad = (transpose(X) * (h_theta_x - y)) / m;

% =============================================================

end
