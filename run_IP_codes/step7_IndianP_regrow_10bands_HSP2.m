clear; pack; clc;

load Indian_pines_corrected.mat;
nChan = size(IndianP_corrected,3);
X_total = reshape(IndianP_corrected,[],nChan);
X_total = X_total';
clear IndianP_corrected;

load dataIndianP_10Bands.mat;
load IndianP_train_test.mat TrainingPointsCell;
load IndianPHSPFineT1.mat stAEOptThetaHSPNew1 inputSize hiddenSizeL3 numClasses netconfigHSP1;
% inputSize = 200;
% hiddenSizeL1 = 160;    % Layer 1 Hidden Size
% hiddenSizeL2 = 120;    % Layer 2 Hidden Size
% hiddenSizeL3 = 600; 
%  numClasses = 16;
 
 ind_train = [];
 trainLabels = [];
%  for k=1:numClasses
%      ind_train = [ind_train;fullTrainingPointsCell{k,1}];
%      trainLabels = [ trainLabels;k*ones(length(fullTrainingPointsCell{k,1}),1)];
%  end
 
  for k=1:numClasses
     ind_train = [ind_train;TrainingPointsCell{k,1}];
     trainLabels = [ trainLabels;k*ones(length(TrainingPointsCell{k,1}),1)];
 end
 
[nRow, nCol, nSli] = size(im_IP10);
[row,col] = ind2sub([nRow, nCol],ind_train);

f = im_IP10;
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
    [pred] = stackedAEPredict3h(stAEOptThetaHSPNew1, inputSize, hiddenSizeL3, ...
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
% save 'regiongrowData2_15.mat'  regiongrowCELL1;
save IndianP_regrow10BHSP2.mat regiongrowCELL1;
