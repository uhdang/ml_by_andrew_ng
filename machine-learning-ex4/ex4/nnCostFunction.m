function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% ------------ Part 1

% printf("theta1 size: %d %d\n", size(Theta1)); 		% 25 X 401
% printf("theta2 size: %d %d\n", size(Theta2)); 		% 10 X 26
% printf("nn_params: %d %d\n", size(nn_params)); 		% 10285 X 1
% printf("input layer size: %d\n", input_layer_size);	% 400
% printf("hidden layer size: %d\n", hidden_layer_size); % 25
% printf("y size: %d %d\n", size(y)); 					% 5000 X 1
% printf("x size: %d %d\n", size(X));					% 5000 X 400
% printf("num_labels: %d %d\n", num_labels); 			% 10

% X with one more layer
a_1 = [ones(size(X, 1), 1) X]; 				% 5000X401
% printf("a_1: %d %d\n", size(a_1));

% Theta_1 = 25X401
z_2 = a_1 * Theta1'; 						% 5000 X 25
a_2 = sigmoid(z_2);
a_2 = [ones(size(a_2,1),1) a_2]; 			% 5000 X 26
% Theta_2 = 10X26
z_3 = a_2 * Theta2'; 						% 5000 X 10
a_3 = sigmoid(z_3);
h_x = a_3;

% a_3: 5000 X 10 , y: 5000 X 1
% !!Important
% Currently y lays out 1-10 values.
% This nees to be expressed in vectors to express 1-10 i.e. [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]
y = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels); % 5000 X 10


% Vectorization vs element-wise multiplication
% There are two ways to get the value J

% 1) Vectorization

foo = (log(h_x)' * -y) - (log(1-h_x)' * (1-y));
% Need to get only "diagonal value" for one-to-one mathing column multiplication
foo = diag(foo);

% (5000 X 10)' * 5000 X 10
% This "foo" gives 10 X 10 matrix. Since we are only interested in one-to-one columns matching values,
% We need to get diagonal value with diag(foo)

% 2) Element-wise multiplication

% foo = -y .* log(h_x) - ((1-y) .* log(1-h_x));
% foo = sum(foo);

% (5000 X 10) .* (5000 X 10)

% This "foo" gives 1 X 10 matrix


% == Regularization ==
% Theta1 - 25 X 401
% Theta2 - 10 X 26

theta1_reg = sum(sum(Theta2(:,2:end) .* Theta2(:,2:end)));
theta2_reg = sum(sum(Theta1(:,2:end) .* Theta1(:,2:end)));

reg = (lambda / (2 * m)) * (theta1_reg + theta2_reg);

J = sum(foo) / m + reg;


% -------------- Part 2
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.


% Attempt 1

reg_grad_2 = (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
reg_grad_1 = (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];

D_2 = zeros(size(Theta2));						% 10 X 26
D_1 = zeros(size(Theta1));						% 26 X 401

for i = 1:m
	a_1i = a_1(i,:);								% 1 X 401
	a_2i = a_2(i,:); 								% 1 X 26
	a_3i = a_3(i,:);								% 1 X 10

	d_3 = a_3i - y(i,:);							% 1 X 10
	d_2 = (d_3 * Theta2) .* (a_2i .* (1-a_2i));		% 1 X 26

	D_2 = D_2 + (d_3' * a_2i);
	D_1 = D_1 + (d_2(2:end)' * a_1i);
endfor

Theta2_grad = 1/m * D_2 + reg_grad_2;
Theta1_grad = 1/m * D_1 + reg_grad_1;;



% WHY??? NOT WORKING ???Attempt 0 - Vectorization
% Need to figure out delta for each layer

% d_3 = a_3 - y; 								% 5000 X 10
% d_2 = (d_3 * Theta2) .* (a_2 .* (1-a_2));	% 5000 X 26
% printf("d_3 size: %d %d\n", size(d_3));
% printf("d_2 size: %d %d\n", size(d_2));

% printf("a_2 size: %d %d\n", size(a_2));
% printf("a_1 size: %d %d\n", size(a_1));
% printf("Before - T2_grad size: %d %d\n", size(Theta2_grad));
% printf("Before - T1_grad size: %d %d\n", size(Theta1_grad));

% Theta2_grad = 1/m * (d_3' * a_2);					% 10 X 26
% Theta1_grad = 1/m * (d_2' * a_1);					% 26 X 401


% printf("After - T2_grad size: %d %d\n", size(Theta2_grad));
% printf("After - T1_grad size: %d %d\n", size(Theta1_grad));



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
% printf("grad size: %d %d\n", size(grad));


end
