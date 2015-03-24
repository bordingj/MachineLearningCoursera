function [J, grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda)
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
y_mat = zeros(num_labels,m);

for i=1:m
    yi =zeros(num_labels,1); yi(y(i)) = 1;
    y_mat(:,i) = yi;
end
%forward propagation
a1 = [ones(m,1), X];
z2 = a1*Theta1';
a2 = [ones(m,1), sigmoid(z2)];
z3 = a2*Theta2';
a3 = sigmoid(z3);
%computing cost
J = -1/m*sum(sum(log(a3').*y_mat+log(1-a3').*(1-y_mat)));
%regularization
J = J+lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

%back propagation
delta_3 = a3-y_mat';
delta_2 = delta_3*Theta2.*sigmoidGradient([ones(m,1) z2]);
delta_2 = delta_2(:,2:end);
Theta1_grad = 1/m*delta_2'*a1;
Theta2_grad = 1/m*delta_3'*a2;
%regularization
Theta1_grad = [Theta1_grad(:,1), Theta1_grad(:,2:end)+lambda/m * Theta1(:,2:end)];
Theta2_grad = [Theta2_grad(:,1), Theta2_grad(:,2:end)+lambda/m * Theta2(:,2:end)];
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
