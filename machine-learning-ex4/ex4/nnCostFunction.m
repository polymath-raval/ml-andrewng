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
a1 = X;
a2 = ( 1 + exp( -1 * [ones(m,1) a1] * transpose(Theta1) ) ) .^ -1;
a3 = ( 1 + exp( -1 * [ones(m,1) a2] * transpose(Theta2) ) ) .^ -1;


for c=1:num_labels
    y1 = (y == c);
    h =  a3(:,c);
    J = J + sum((y1 .* log(h)) + ((1 - y1) .* log(1-h)));
end
J = -1 * J / m;
sqr_t1 = Theta1(:,2:size(Theta1,2)) .^ 2;
sqr_t2 = Theta2(:,2:size(Theta2,2)) .^ 2;
regularized_comp = sum(sqr_t1(:)) + sum(sqr_t2(:));
J = J + lambda * 0.5 * regularized_comp / m;
% -------------------------------------------------------------

% =========================================================================

y1 = zeros(m,num_labels);
for c=1:num_labels
    y1(:,c) = (y==c);
end    
for t=1:m
    a1 = X(t,:);
    a2 = (1 + exp(-1 * [1 a1] * transpose(Theta1))) .^ -1;
    del3 = ((1 + exp(-1 * [1 a2] * transpose(Theta2))) .^ -1) - y1(t,:);
    del2 = del3 * Theta2(:,2:size(Theta2,2)) .* (a2 .* (1-a2));
    Theta1_grad = Theta1_grad + transpose(del2) * [1 a1];
    Theta2_grad = Theta2_grad + transpose(del3) * [1 a2];
end
Theta1_grad = (Theta1_grad/m) + lambda / m * [zeros(size(Theta1,1),1) Theta1(:,2:size(Theta1,2))];
Theta2_grad = (Theta2_grad/m)  + lambda / m * [zeros(size(Theta2,1),1) Theta2(:,2:size(Theta2,2))];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
