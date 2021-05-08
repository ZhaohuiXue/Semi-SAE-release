function [] = IndianPSampIe()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
load Indian_pines_gt.mat;

[M, N] = size(IndianP_gt);
numClasses = max(IndianP_gt(:));

ind_label = find(IndianP_gt);
X_labels = IndianP_gt(ind_label);

TrainingPointsCell = cell(numClasses,1);
TestPointsCell = cell(numClasses,1);
for k=1:numClasses
    temp = find(X_labels==k);
    temp = ind_label(temp);
    ind = randperm(length(temp));
    temp = temp(ind);
    c1 = round(0.1*length(temp));% 0.1 for train
    TrainingPointsCell{k,1} = temp(1:c1);
    TestPointsCell{k,1} = temp(c1+1:end);
    
end
save 'IndianP_train_test.mat'  TrainingPointsCell  TestPointsCell; 

