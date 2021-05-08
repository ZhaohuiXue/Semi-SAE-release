clear,pack,clc
%% 
inputSize = 103;
hiddenSizeL1 = 80;    % Layer 1 Hidden Size
hiddenSizeL2 = 60;    % Layer 2 Hidden Size
hiddenSizeL3 = 300;    % Layer 3 Hidden Size
maxIter = 15000;       % 精调迭代次数
%% 
load PaviaU_train_test TrainingPointsCell TestPointsCell;
load PaviaU_HSP.mat sae1OptTheta sae2OptTheta sae3OptTheta; 
load PaviaU.mat PaviaU;
%========================================%
nChan = size(PaviaU,3);
X_total = reshape(PaviaU,[],nChan);
X_total = X_total';
clear PaviaU;
%========================================%
 numClasses = 9;
 ind_train = [];
 trainLabels = [];
 
 for k=1:numClasses
     ind_train = [ind_train;TrainingPointsCell{k,1}];
     trainLabels = [ trainLabels;k*ones(length(TrainingPointsCell{k,1}),1)];
 end
 %=======================================%

sparsityParam = 0.15;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-4;         % weight decay parameter       
beta = 4;              % weight of sparsity penalty term     
 
 %======================================%
X = X_total(:,ind_train);%training samples
[X] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                     inputSize, X);
[X] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                     hiddenSizeL1, X);
[X] = feedForwardAutoencoder(sae3OptTheta, hiddenSizeL3, ...
                                    hiddenSizeL2, X);
 %=======================================%
 
saeSoftmaxTheta = 0.05 * randn(hiddenSizeL3 * numClasses, 1);
options.maxIter = 2000;
softmaxModel = softmaxTrain(hiddenSizeL3, numClasses, lambda, ...
                            X,trainLabels, options);

saeSoftmaxOptTheta = softmaxModel.optTheta(:);

clear saeSoftmaxTheta;

%%======================================================================
%%  Finetune  model
% Initialize the stack using the parameters learned
stack = cell(3,1);

%第一层自动编码机的w1，b1
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
%第二层自动编码机的w2，b2
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);
%第三层自动编码机的w3，b3
stack{3}.w = reshape(sae3OptTheta(1:hiddenSizeL3*hiddenSizeL2), ...
                     hiddenSizeL3, hiddenSizeL2);
stack{3}.b = sae3OptTheta(2*hiddenSizeL3*hiddenSizeL2+1:2*hiddenSizeL3*hiddenSizeL2+hiddenSizeL3);

% Initialize the parameters for the deep model
[stackparams, netconfigHSP1] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

%% -------------------------------------------------------
lambda = 1e-4;

% addpath minFunc/
options.Method = 'lbfgs'; 
options.maxIter = maxIter;	 
options.display = 'on';
options.maxFunEvals = 20000;
[stAEOptThetaHSP1, cost] = minFunc( @(p) stackedAECost3h(p, inputSize, hiddenSizeL3, ...
                                              numClasses, netconfigHSP1, lambda, ...
                                              X_total(:,ind_train),  trainLabels), stackedAETheta, options);
save PaviaUHSPFineT1.mat stAEOptThetaHSP1 inputSize hiddenSizeL3 numClasses netconfigHSP1;
%%======================================================================
%%  Test 
clear TrainingPointsCell sae1OptTheta sae2OptTheta sae3OptTheta  ind_train trainLabels stack

ind_test = [];
 testLabels = [];
 for k=1:numClasses
     ind_test = [ind_test;TestPointsCell{k,1}];
     testLabels = [ testLabels;k*ones(length(TestPointsCell{k,1}),1)];
 end
[pred] = stackedAEPredict3h(stackedAETheta, inputSize, hiddenSizeL3, ...
                          numClasses, netconfigHSP1, X_total(:,ind_test));

acc1 = mean(testLabels(:) == pred(:));
[pred] = stackedAEPredict3h(stAEOptThetaHSP1, inputSize, hiddenSizeL3, ...
                          numClasses, netconfigHSP1, X_total(:,ind_test));

acc2 = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc2 * 100);

