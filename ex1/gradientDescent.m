function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    tt = theta;
    C = X * theta - y;
    
    for j = 1:2
      tt(j) = theta(j) - sum(C .* X(:,j))*alpha/m;
    end

    theta = tt;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    % check if cost is lower than previous iteration
    if (iter > 1 && J_history(iter) >= J_history(iter-1)) 
      disp("ERROR, cost increased during gradient descent!");
    endif
end

end
