clear,pack,clc
load dataIndianP_Patch.mat dataIndianPPatch; 
nChan = size(dataIndianPPatch,3);
if nChan>1
    error('The input data must be a two-dimensional array!');
end
inputSize = 330;
hiddenSizeL1 = 200;    % Layer 1 Hidden Size
hiddenSizeL2 = 120;    % Layer 2 Hidden Size
hiddenSizeL3 = 600;    % Layer 3 Hidden Size
sparsityParam = 0.15;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-4;         % weight decay parameter       
beta = 4;              % weight of sparsity penalty term     
 
 

%  Randomly initialize the parameters
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 1200;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
options.Corr = 10;

[sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSizeL1, ...
                                   lambda, sparsityParam, ...
                                   beta, dataIndianPPatch), ...
                                   sae1Theta, options);

save IndianP_SPF.mat sae1OptTheta
% -------------------------------------------------------------------------
%%======================================================================

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, dataIndianPPatch);
clear  sae1Theta;

sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);

options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 800;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   hiddenSizeL1, hiddenSizeL2, ...
                                   lambda, sparsityParam, ...
                                   beta, sae1Features), ...
                                   sae2Theta, options);

 save IndianP_SPF.mat sae2OptTheta -append;                             

[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);
clear sae1Features sae2Theta;
%  Randomly initialize the parameters
sae3Theta = initializeParameters(hiddenSizeL3, hiddenSizeL2);
% -------------------------------------------------------------------------

%%======================================================================
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 600;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

[sae3OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   hiddenSizeL2, hiddenSizeL3, ...
                                   lambda, sparsityParam, ...
                                   beta, sae2Features), ...
                                   sae3Theta, options);


save IndianP_SPF.mat sae3OptTheta inputSize hiddenSizeL1  -append;  
save IndianP_SPF.mat hiddenSizeL2  hiddenSizeL3 -append;
