function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



unregularized_j = (sum(((X * theta) - y) .^ 2) * 0.5 / m);
regularized_j = (sum(theta(2:size(theta,1)) .^ 2) * lambda * 0.5 / m );
J = unregularized_j + regularized_j;
unregularized_grad = (transpose(X) * (X * theta - y)) / m;
regularized_grad = [zeros(1,1);theta(2:size(theta,1)) * lambda /m];
grad = unregularized_grad + regularized_grad;







% =========================================================================

grad = grad(:);

end
