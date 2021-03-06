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
output_layer_size = 10;
         
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

% add bias column
X = [ones(m,1) X]; % (5000,401)
a1 = X; % (5000, 401)

% layer2 (hidden_layer)
a2 = sigmoid(a1 * Theta1'); % (5000, 25)
a2 = [ones(m,1) a2]; % (5000, 26)

% layer3 (output_layer)
a3 = sigmoid(a2 * Theta2'); % (5000, 10)
h = a3;

% first calculate J for y=0
J = sum(-log(1 - h)(:));

% then re-add the y=1 values
for i = 1:m
  ah = h(i, y(i));
  J = J + log(1-ah) - log(ah);
end

J = J / m;

% add regularization - generate non-bias thetas first
t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));
J = J + lambda/(2*m)*(sum((t1.^2)(:)) + sum((t2.^2)(:)));

% -------------------------------------------------------------
% back propagation

% create expanded y
ye = zeros(m, num_labels); % (5000, 10)
for i = 1:m
  ye(i,y(i))=1;
end
% this could be usable above to replace the for -- but at the cost of having to 
% create a lot of matrices, which, in the end, is probably slower than an O(n) 
% for cycle with a negligible body

d3 = a3 - ye; % (5000, 10)

% a2 is already sigmoid so we just calculate sigmoidGradient on the fly
d2 = (d3 * Theta2) .* (a2 .* (1 - a2)); % (5000, 26)
% remove bias 
d2 = d2(:,2:end); % (5000, 25)

Theta2_grad = Theta2_grad + d3' * a2; % (10, 26)
Theta1_grad = Theta1_grad + d2' * a1; % (25, 401)

% add regularization (except for bias)
t1 = Theta1;
t2 = Theta2;
t1(:,1) = 0;
t2(:,1) = 0;
Theta1_grad = (Theta1_grad + lambda * t1)/m;
Theta2_grad = (Theta2_grad + lambda * t2)/m;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
