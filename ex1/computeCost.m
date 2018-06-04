function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
[m1,n1]=size(X);
[m2,n2]=size(theta);
if m2~=n1 && n2~=1
   error('Dimension mismatch or undesired dimension');
end
predictions=zeros(m1,1);
if n2~=1
 predictions=X*theta';
else
 predictions=X*theta;
end

J = sum((predictions-y).^2)/(2*m);



end
