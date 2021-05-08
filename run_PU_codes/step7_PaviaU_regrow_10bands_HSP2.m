clear;clc;
%% 
%% 
load dataPaviaU_10Bands.mat;
load PaviaU_train_test TrainingPointsCell TestPointsCell;
load PaviaU.mat PaviaU;
load PaviaUHSPFineT1.mat stAEOptThetaHSP1 inputSize hiddenSizeL3 numClasses netconfigHSP1;

nChan = size(PaviaU,3);
X_total = reshape(PaviaU,[],nChan);
X_total = X_total';

 ind_train = [];
 trainLabels = [];
 
  for k=1:numClasses
     ind_train = [ind_train;TrainingPointsCell{k,1}];
     trainLabels = [ trainLabels;k*ones(length(TrainingPointsCell{k,1}),1)];
 end
 
[nRow, nCol, nSli] = size(im_PU10);
[row,col] = ind2sub([nRow, nCol],ind_train);

f = im_PU10;
for k=1:nSli
    f(:,:,k) = medfilt2(f(:,:,k));
end
regiongrowCELL1 = cell(numClasses ,1);
for k=1:length(ind_train)
    ind = regiongrowHHUZhou(f,[row(k),col(k)]);
    k_label = trainLabels(k);
    temp = regiongrowCELL1{k_label,1};
    temp = [temp;ind];
    regiongrowCELL1{k_label,1} = temp;
end

for k=1:numClasses
    regiongrowCELL1{k,1} = unique(regiongrowCELL1{k,1});
end

for kkk=1:numClasses
    ind = regiongrowCELL1{kkk,1};
    X = X_total(:,ind);
    [pred] = stackedAEPredict3h(stAEOptThetaHSP1, inputSize, hiddenSizeL3, ...
                          numClasses,  netconfigHSP1, X);
    
    indb = find(pred==kkk);
    regiongrowCELL1{kkk,1} = ind(indb);
end

 for k=1:numClasses
     regiongrowCELL1{k,1} = [regiongrowCELL1{k,1};TrainingPointsCell{k,1}];
 end

for k=1:numClasses
    regiongrowCELL1{k,1} = unique(regiongrowCELL1{k,1});
end

save PaviaU_regrow10BHSP2.mat regiongrowCELL1;
