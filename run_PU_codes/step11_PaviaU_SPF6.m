clear,pack,clc
inputSize = 330;
hiddenSizeL1 = 240;    % Layer 1 Hidden Size
hiddenSizeL2 = 140;    % Layer 2 Hidden Size
hiddenSizeL3 = 600;    % Layer 3 Hidden Size
maxIter = 15000;       % 精调迭代次数
%% 
load PaviaU_regrow10BSPF4.mat regiongrowCELL2;
load PaviaU_SPF.mat sae1OptTheta sae2OptTheta sae3OptTheta;
load dataPaviaU_Patch.mat;
M = 610; N = 340;
numClasses = 9;
ind_train2 = [];
trainLabels2 = [];
for k=1:numClasses
    temp = regiongrowCELL2{k,1};
    ind_train2 = [ind_train2;temp(:)];
    trainLabels2 = [trainLabels2;k*ones(length(temp),1)];
end
% extraInd = [];
% extraLabels = [];
% 
% temp = randsample(regiongrowCELL2{1,1},100,1);
% extraInd = [extraInd;temp];
% extraLabels = [extraLabels;ones(100,1)];
% 
% temp = randsample(regiongrowCELL2{7,1},100,1);
% extraInd = [extraInd;temp];
% extraLabels = [extraLabels;7*ones(100,1)];
% 
% temp = randsample(regiongrowCELL2{9,1},100,1);
% extraInd = [extraInd;temp];
% extraLabels = [extraLabels;9*ones(100,1)];

X = dataPaviaUPatch(:,ind_train2);
% XB = dataPaviaUPatch(:,extraInd);
% XB = XB + 0.01*randn(size(XB));
% X = [X,XB];
%  trainLabels2 = [ trainLabels2;extraLabels];

sparsityParam = 0.15;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-4;         % weight decay parameter       
beta = 4;              % weight of sparsity penalty term       

[X] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, X);
[X] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, X);
[X] = feedForwardAutoencoder(sae3OptTheta, hiddenSizeL3, ...
                                        hiddenSizeL2, X);

saeSoftmaxTheta = 0.05 * randn(hiddenSizeL3 * numClasses, 1);

options.maxIter = 2000;
softmaxModel = softmaxTrain(hiddenSizeL3, numClasses, lambda, ...
                            X,trainLabels2, options);

saeSoftmaxOptTheta = softmaxModel.optTheta(:);

%%======================================================================
%%  Finetune model

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
[stackparams, netconfigSPF] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

X = dataPaviaUPatch(:,ind_train2);
% X = [X,XB];

lambda = 1e-4;

% addpath minFunc/
options.Method = 'lbfgs'; 
options.maxIter = maxIter;	 
options.display = 'on';
options.maxFunEvals = 100000;
[stAEOptThetaSPF2, cost] = minFunc( @(p) stackedAECost3h(p, inputSize, hiddenSizeL3, ...
                                              numClasses, netconfigSPF, lambda, ...
                                              X, trainLabels2), stackedAETheta, options);
%save( FineSavePath,'stAEOptThetaSPF2','inputSize','hiddenSizeL3','numClasses','netconfigSPF');
%%======================================================================
%%  Test 
clear TrainingPointsCell sae1OptTheta sae2OptTheta sae3OptTheta  ind_train trainLabels stack

load PaviaU_train_test.mat TestPointsCell;
ind_test = [];
 testLabels = [];
 for k=1:numClasses
     ind_test = [ind_test;TestPointsCell{k,1}];
     testLabels = [ testLabels;k*ones(length(TestPointsCell{k,1}),1)];
 end

[pred] = stackedAEPredict3h(stAEOptThetaSPF2, inputSize, hiddenSizeL3, ...
                          numClasses, netconfigSPF, dataPaviaUPatch(:,ind_test));

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

[pred] = stackedAEPredict3h(stAEOptThetaSPF2, inputSize, hiddenSizeL3, ...
                          numClasses,netconfigSPF, dataPaviaUPatch);
classDataSPF = reshape(pred,M,N);

[probDataSPF] = stackedAEPredict3hProb(stAEOptThetaSPF2, inputSize, hiddenSizeL3, ...
                          numClasses,netconfigSPF, dataPaviaUPatch);
save PaviaUSPFFineT6.mat classDataSPF probDataSPF;

