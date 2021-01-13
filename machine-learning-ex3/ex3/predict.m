function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% printf("Theta1 size - %d %d, Theta2 size - %d %d, X size - %d %d\n", size(Theta1), size(Theta2), size(X));

% Add a column to X as a "bias unit"
X = [ones(size(X, 1), 1) X];
% printf("X after adding a row of ones - %d %d\n", size(X));
% Second Layer - a^2
z_2 = X * Theta1';
a_2 = sigmoid(z_2);
% printf("a_2 size - %d %d\n", size(a_2));

% Third layer - a^3 = h_theta(x)
% Add a column to a_2 as a "bias unit"
a_2 = [ones(size(a_2, 1), 1) a_2];
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);

[v, v_idx] = max(a_3,[], 2);
p = v_idx;

% =========================================================================


end