function [] = PaviaUSample()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
load PaviaU_gt.mat;

[M, N] = size(PaviaU_gt);
numClasses = max(PaviaU_gt(:));

ind_label = find(PaviaU_gt);
X_labels = PaviaU_gt(ind_label);

TrainingPointsCell = cell(numClasses,1);
TestPointsCell = cell(numClasses,1);
for k=1:numClasses
    temp = find(X_labels==k);
    temp = ind_label(temp);
    ind = randperm(length(temp));
    temp = temp(ind);
    c1 = round(0.05*length(temp));% 0.05 for train
    TrainingPointsCell{k,1} = temp(1:c1);
    TestPointsCell{k,1} = temp(c1+1:end);
    
end
save 'PaviaU_train_test.mat'  TrainingPointsCell  TestPointsCell; 

