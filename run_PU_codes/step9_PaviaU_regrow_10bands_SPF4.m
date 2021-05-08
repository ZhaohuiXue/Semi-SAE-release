clear;clc;
%% 
load PaviaUSPFFineT3.mat stAEOptThetaSPF1 inputSize hiddenSizeL3 numClasses netconfigSPF;
%% 
load dataPaviaU_10Bands.mat im_PU10;
load PaviaU_train_test TrainingPointsCell TestPointsCell;
load dataPaviaU_Patch.mat;

[nRow, nCol, nSli] = size(im_PU10);

f = im_PU10;
for k=1:nSli
    f(:,:,k) = medfilt2(f(:,:,k));
end
regiongrowCELL2 = cell(numClasses ,1);
for kk=1:numClasses
    [row,col] = ind2sub([nRow, nCol],TrainingPointsCell{kk,1});
    for k=1:length(row)
        ind = regiongrowHHUZhou(f,[row(k),col(k)]);
        regiongrowCELL2{kk,1} = [regiongrowCELL2{kk,1};ind];
    end
end

for k=1:numClasses
    regiongrowCELL2{k,1} = unique(regiongrowCELL2{k,1});
end

for kkk=1:numClasses
    ind = regiongrowCELL2{kkk,1};
%     X = X_total(:,ind);
    X = dataPaviaUPatch(:,ind);
    [pred] = stackedAEPredict3h(stAEOptThetaSPF1, inputSize, hiddenSizeL3, ...
                          numClasses, netconfigSPF, X);
%     [pred] = stackedAEPredict3h(stAEOptThetaHSPNew1, inputSize, hiddenSizeL3, ...
%                           numClasses,  netconfigHSP1, X);
    
    indb = find(pred==kkk);
    regiongrowCELL2{kkk,1} = ind(indb);
end

for k=1:numClasses
    regiongrowCELL2{k,1} = [regiongrowCELL2{k,1};TrainingPointsCell{k,1}];
end

for k=1:numClasses
    regiongrowCELL2{k,1} = unique(regiongrowCELL2{k,1});
end
% save 'regiongrowData2_15.mat'  regiongrowCELL2 -append;
save PaviaU_regrow10BSPF4.mat regiongrowCELL2;
