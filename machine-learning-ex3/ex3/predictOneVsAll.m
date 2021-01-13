function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% printf("all_theta size - %d %d | ex - %d\n", size(all_theta), all_theta(1));
% printf("m - %d\n", m);
% printf("num_labels - %d\n", num_labels);
% printf("p size - %d %d | ex - %d\n", size(p), p(1));
% printf("X size - %d %d | ex - %d\n", size(X), X(1));
% [max_v, v_idx] = max(X, [], 2);
% printf("size max_v - %d %d | ex - %d\n", size(max_v), max_v(1));
% printf("v_idx - %d %d || first element: %d\n", size(v_idx), v_idx(1));

% How do i get the probability that it belongs to each class for each input ?% What is the input ?

% all_theta - 10 x 401
% X - 5000 X 401
z = X * all_theta'; % 5000 X 10
% size(z)
% z = all_theta * X'; % 10 X 5000
prob = sigmoid(z);
% size(prob)

% Get the index of max value and apply it to the class (num_labels)
[v, idx] = max(prob, [], 2);
p = idx;






% =========================================================================


end
