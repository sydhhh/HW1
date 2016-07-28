function [test_targets, a_star] = my_svm(train_patterns, train_targets, test_patterns, params)

% Classify using (a very simple implementation of) the support vector machine algorithm
% 
% Inputs:
% 	train_patterns	- Train patterns
%	train_targets	- Train targets
%   test_patterns   - Test  patterns
%	params	        - [kernel, kernel parameter, solver type, Slack]
%                     Kernel can be one of: Gauss, RBF (Same as Gauss), Poly, Sigmoid, or Linear
%                     The kernel parameters are:
%                       RBF kernel  - Gaussian width (One parameter)
%                       Poly kernel - Polynomial degree
%                       Sigmoid     - The slope and constant of the sigmoid (in the format [1 2], with no separating commas)
%					    Linear		- None needed
%                     Solver type can be one of: Perceptron, Quadprog, Lagrangian, SEQ
%
% Outputs
%	test_targets	- Predicted targets
%	a			    - SVM coeficients
%
% Note: The number of support vectors found will usually be larger than is actually 
% needed because the two first solvers are approximate.

[Dim, Nf]       = size(train_patterns);
Dim             = Dim + 1;
train_patterns(Dim,:) = ones(1,Nf);
test_patterns(Dim,:)  = ones(1, size(test_patterns,2));

if (length(unique(train_targets)) == 2)
    z   = 2*(train_targets>0) - 1; 
else
    z   = train_targets;
end

%Get kernel parameters
[kernel, ker_param, solver, slack] = process_params(params);

%Transform the input patterns
y	= zeros(Nf);
switch kernel
case {'Gauss','RBF'},
   for i = 1:Nf,
      y(:,i)    = exp(-sum((train_patterns-train_patterns(:,i)*ones(1,Nf)).^2)'/(2*ker_param^2));
   end
case {'Poly', 'Linear'}
   if strcmp(kernel, 'Linear')
      ker_param = 1;
   end
   for i = 1:Nf,
      y(:,i) = (train_patterns'*train_patterns(:,i) + 1).^ker_param;
   end
otherwise
   error('Unknown kernel. Can be Gauss, Linear, Poly, or Sigmoid.')
end

%Find the SVM coefficients
switch solver
case 'Quadprog'
   %Quadratic programming
   alpha_star	= quadprog(diag(z)*y'*y*diag(z), -ones(1, Nf), zeros(1, Nf), 1, z, 0, zeros(1, Nf), slack*ones(1, Nf))';
   a_star		= (alpha_star.*z)*y';
   
   %Find the bias
   sv_for_bias  = find((alpha_star > 0) & (alpha_star < slack - 0.001*slack));
   %sv_for_bias  = find((alpha_star > 0.001*slack) & (alpha_star < slack - 0.001*slack));
   if isempty(sv_for_bias),
       bias     = 0;
   else
	   B        = z(sv_for_bias) - a_star(sv_for_bias);
       bias     = mean(B);
   end
   
   sv           = find(alpha_star > 0);
   %sv           = find(alpha_star > 0.001*slack);
otherwise
   error('Unknown solver. Can be either Quadprog or Perceptron')
end

%Find support verctors
Nsv	    = length(sv);
if isempty(sv),
   error('No support vectors found');
else
   disp(['Found ' num2str(Nsv) ' support vectors'])
end

%Margin
b	= 1/sqrt(sum(a_star.^2));
disp(['The margin is ' num2str(b)])

%Classify test patterns
N   = size(test_patterns, 2);
y   = zeros(1, N);

for i = 1:Nsv,
    switch kernel,
    case {'Gauss','RBF'},
        y		    = y + a_star(sv(i)) * exp(-sum((test_patterns-train_patterns(:,sv(i))*ones(1,N)).^2)'/(2*ker_param^2))';
    case {'Poly', 'Linear'}
        y		    = y + a_star(sv(i)) * (test_patterns'*train_patterns(:,sv(i))+1)'.^ker_param;
    end
end

test_targets = y + bias;

if (length(unique(train_targets)) == 2)
    test_targets = test_targets > 0;
end
%%%%%%%%%%%%%