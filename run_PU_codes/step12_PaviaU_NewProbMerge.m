clear;pack;clc;
load PaviaUHSPFineT5.mat;
load PaviaUSPFFineT6.mat;
load PaviaU_train_test.mat;
%TrainingPointsCell, TestPointsCell

M = 610; N = 340; Tregion = 4;
numClasses = uint8(9);

imt1 = single(classDataHSP);
clear classDataHSP;

imt2 = single(classDataSPF);
clear classDataSPF;

% imt1 = single(classDataSPFcye);
% clear classDataSPFcye;
% [~,pred]=max(probDataSPF);
% imt2 = single(reshape(pred,M,N));
BW = imt1 == imt2;
% [~,pred]=max(probDataHSP + probDataSPF);
% imt3 = single(reshape(pred,M,N));
% imt3(BW) = imt1(BW);
% imt1 = imt3;
% clear imt3;

for k=1:numClasses
    ind = TrainingPointsCell{k,1};
    imt1(ind) = k;
    BW(ind) = true;
end
% The class information of the points labelled in BW cannot be changed.

% probTemp = probDataHSP + probDataSPF;
% [~,pred]=max(probTemp);
% imt1 = single(reshape(pred,M,N));
% imt2 = imt1;
% % [~,pred]=max(probHSP2_4);
% % [~,pred2]=max(probSPF2_4);
% % ind1 = find(pred==pred2);
% % ind2 = find(pred(ind_train3)==trainLabels3');
% % mask = false(M,N);
% % mask(intersect(ind1,ind_train3(ind2))) = true;
% % imt1(mask) = single(pred(mask(:)));
% % imt2 = imt1;
% % imt1(mask) = pred(mask(:));
% % imt2 = imt1;
% % clear pred pred2 probTemp ind1 ind2;

NeighB8 = single([-M-1, -M, -M+1,1,M+1,M,M-1,-1]);
% P1 = probDataHSP + 0.50*probDataSPF;
P1 = probDataHSP + 0.65*probDataSPF;

P1 = bsxfun(@rdivide,P1,sum(P1));
P1 = bsxfun(@minus,sum(P1),P1);
% P1 = single(bsxfun(@minus,ones(1,size(probDataHSP,2)),probDataHSP)); 
clear probDataHSP;
% P2 = single(bsxfun(@minus,ones(1,size(probDataSPF,2)),probDataSPF)); 
clear probDataSPF;
% PT = P1;
% P1 = P2;
% P2 = PT;
% belta12 = single(0.35);
belta_boundary = single(0.4);
beltaN = single(0.8);
iterations = 10;

tempV = zeros(numClasses,1,'single');

singleIND = single(reshape(1:M*N,M,N));
% centralPartIND = zeros((M-2)*(N-2),3,'single');
temp1 = singleIND(2:(end-1),2:(end-1));
centralPartIND = (temp1(:))';
% centralPartIND(:,2:3) = single(ind2sub([M,N],centralPartIND(:,1)));
temp1 = P1(:,centralPartIND);

for kkk=1:iterations
    %%% First consider the 4 corner points.
    if ~BW(1,1)
        tempV = P1(:,singleIND(1,1));
        for k=1:numClasses
            tempV(k) = tempV(k) + belta_boundary*(imt1(1,2)~=k) + ...
                belta_boundary*(imt1(2,1)~=k) + belta_boundary*(imt1(2,2)~=k);
        end
        [~,ind] = min(tempV);
        imt2(1,1) = single(ind);
    end
    
    if ~BW(M,1)
        tempV = P1(:,singleIND(M,1));
        for k=1:numClasses
            tempV(k) = tempV(k) + belta_boundary*(imt1(M-1,1)~=k) + ...
                belta_boundary*(imt1(M-1,2)~=k) + belta_boundary*(imt1(M,2)~=k);
        end
        [~,ind] = min(tempV);
        imt2(M,1) = single(ind);
    end

    if ~BW(1,N)
        tempV = P1(:,singleIND(1,N));
        for k=1:numClasses
            tempV(k) = tempV(k) + belta_boundary*(imt1(1,N-1)==k) + ...
                belta_boundary*(imt1(2,N-1)~=k) + belta_boundary*(imt1(2,N)~=k);
        end
        [~,ind] = min(tempV);
        imt2(1,N) = single(ind);
    end

    if ~BW(M,N)
        tempV = P1(:,singleIND(M,N));
        for k=1:numClasses
            tempV(k) = tempV(k) + belta_boundary*(imt1(M,N-1)~=k) + ...
                belta_boundary*(imt1(M-1,N-1)~=k) + belta_boundary*(imt1(M-1,N)~=k);
        end
        [~,ind] = min(tempV);
        imt2(M,N) = single(ind);
    end

    %%%--------------------------------------------------------------------------------%%%%
    %===================================================%
    % Then consider 4 boundaries without corners.
    for k1=2:(M-1)
        if ~BW(k1,1)
            tempV = P1(:,singleIND(k1,1));
            for k2=1:numClasses
                tempV(k2) = tempV(k2) + belta_boundary*(imt1(k1-1,1)~=k) + ...
                    belta_boundary*(imt1(k1-1,2)~=k) + belta_boundary*(imt1(k1,2)~=k) +...
                    belta_boundary*(imt1(k1+1,2)~=k) + belta_boundary*(imt1(k1+1,1)~=k);
            end
            [~,ind] = min(tempV);
            imt2(k1,1) = single(ind);
        end
    end

    for k1=2:(M-1)
        if ~BW(k1,N)
            tempV = P1(:,singleIND(k1,N));
            for k2=1:numClasses
                tempV(k2) = tempV(k2) + belta_boundary*(imt1(k1-1,N)~=k) + ...
                    belta_boundary*(imt1(k1-1,N-1)~=k) + belta_boundary*(imt1(k1,N-1)~=k) +...
                    belta_boundary*(imt1(k1+1,N-1)~=k) + belta_boundary*(imt1(k1+1,N)~=k);
            end
            [~,ind] = min(tempV);
            imt2(k1,N) = single(ind);
        end
    end

    for k1=2:(N-1)
        if ~BW(1,k1)
            tempV = P1(:,singleIND(1,k1));
            for k2=1:numClasses
                tempV(k2) = tempV(k2) + belta_boundary*(imt1(1,k1-1)~=k) +...
                    belta_boundary*(imt1(2,k1-1)~=k) + belta_boundary*(imt1(2,k1)~=k) + ...
                    belta_boundary*(imt1(2,k1+1)~=k) + belta_boundary*(imt1(1,k1+1)~=k);
            end
            [~,ind] = min(tempV);
            imt2(1,k1) = single(ind);
        end
    end

    for k1=2:(N-1)
        if ~BW(M,k1)
            tempV = P1(:,singleIND(M,k1));
            for k2=1:numClasses
                tempV(k2) = tempV(k2) + belta_boundary*(imt1(M,k1-1)~=k) +...
                    belta_boundary*(imt1(M-1,k1-1)~=k) + belta_boundary*(imt1(M-1,k1)~=k) +...
                    belta_boundary*(imt1(M-1,k1+1)~=k) + belta_boundary*(imt1(M,k1+1)~=k);
            end
            [~,ind] = min(tempV);
            imt2(M,k1) = single(ind);
        end
    end

    %%%======================================%%%%
    % The code to deal with the central image is very easy though it is
    % time consuming.
    temp2 = temp1;
    for k=1:numClasses
        for k1=1:8
            tempp = imt1(centralPartIND+NeighB8(k1));
            temp2(k,:) = temp2(k,:) + beltaN*(tempp~=k);
        end
    end
    [~,ind] = min(temp2);
    imt2(centralPartIND) = single(ind);
    imt2(BW) = imt1(BW);   % The labelled pixels are not changed in the process since their confidence is high.
    imt1 = imt2;
%     temp=imt2(ind_lable);
%     sum(temp(ind_test)==X_labels(ind_test))/length(ind_test)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Merge classified regions.
clear temp1 temp2 tempp tempV singleND NeighB8;
% load 'C:\zsg\Indian Pine\zsg_AE_CLASSIFICATION\Data\TenBandsSelected\dataIndianP_10Bands.mat';
load dataPaviaU_10Bands.mat;
% This line needs to be adjusted while the function is carried out on different
% computers.
nBand = size(im_PU10,3);
im_PU10 = reshape(im_PU10,[],nBand);

jj = 0;
PixelIdxList = [];
regionLabels = [];
regionPixelNum = [];
regionMean = [];

for k1=1:numClasses 
    temp = bwconncomp(imt1==k1);
    for k2=1:temp.NumObjects
        jj = jj + 1;
        PixelIdxList{jj,1} = temp.PixelIdxList{1,k2};
        imt2(PixelIdxList{jj,1}) = single(jj);
        regionPixelNum(jj) = numel(temp.PixelIdxList{1,k2});
        if regionPixelNum(jj)>1
            regionMean(jj,:) = mean(im_PU10([PixelIdxList{jj,1}],:));
        else
            regionMean(jj,:) = im_PU10([PixelIdxList{jj,1}],:);
        end
        regionLabels(jj) = k1;
    end
end
ind = find(regionPixelNum<Tregion);
[Am, Al] = regionadjacency(double(imt2));
regionLablled = false(size(Al,1),1);
for k1 = ind
    NeighB = Al{k1,1};
    tempV = [];
    tempNb = [];
    jj = 0;
    for k2 = NeighB
        if ~regionLablled(k2)
            jj = jj + 1;
            tempV(jj) = sum(abs(regionMean(k2) - regionMean(k1)));
            tempNb(jj) = k2;
        end
    end
    if ~isempty(tempV)
        [~,k3] = min(tempV);
        regionLabels(k1) = regionLabels(tempNb(k3));
        regionLablled(k1) = true;
    end
end
imt2(:) = 0;
for k=1:length(regionLabels)
     imt2(PixelIdxList{k,1}) = single(regionLabels(k));
end
imt1 = imt2;


ind_test = [];
 testLabels = [];
 for k=1:numClasses
     ind_test = [ind_test;TestPointsCell{k,1}];
     testLabels = [ testLabels;double(k)*ones(length(TestPointsCell{k,1}),1)];
 end

pred=imt1(ind_test);
mean(pred(:)==testLabels(:))
% load 'C:\zsg\Indian Pine\zsg_AE_CLASSIFICATION\Data\RawData\Indian_pines_gt.mat';
load PaviaU_gt.mat;
% This line needs to be adjusted while the function is carried out on different
% computers.
ind_label = find(IndianP_gt);
imt2(:) = 0;
imt2(ind_label) = imt1(ind_label);
save(sprintf('%s%s%s','PU_final_map_',datestr(now,30),'.mat'),'imt2');