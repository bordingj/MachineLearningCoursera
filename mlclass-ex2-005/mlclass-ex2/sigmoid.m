function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
dim = size(z);
g = zeros(dim);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar)
for i=1:dim(1)
    for j=1:dim(2)
        g(i,j) = 1/(1+exp(-z(i,j)));
    end
end




% =============================================================

end
