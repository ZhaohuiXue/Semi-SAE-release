%For 9x9 sub windows
%EncirclementCell{1,1} contains the central 3x3 window;
% Each of other EncirclementCells contains 8 segments of a  rectangular ring with one pixel width.
top3 = 4; bottom3 = 6; left3 = 4; right3 = 6;
top5 = 3; bottom5 = 7; left5 = 3; right5 = 7;
top7 = 2; bottom7 = 8; left7 = 2; right7 = 8;
top9 = 1; bottom9 = 9; left9 = 1; right9 = 9;
rowCent = 5; colCent = 5;

indPatch = reshape([1:81],9,9);
EncirclementCell = cell(4,1);
temp = indPatch(top3:bottom3,left3:bottom3);
EncirclementCell{1,1} = (temp(:))';

segment8 = zeros(3,2,8);
segment8(:,:,1) = [top5 left5; top5+1 left5;top5 left5+1];
segment8(:,:,2) = [top5 colCent; top5 colCent-1;top5 colCent+1];
segment8(:,:,3) = [top5 right5; top5 right5-1; top5+1 right5];
segment8(:,:,4) = [rowCent right5; rowCent-1 right5; rowCent+1 right5];
segment8(:,:,5) = [bottom5 right5; bottom5-1 right5; bottom5 right5-1];
segment8(:,:,6) = [bottom5 colCent; bottom5 colCent-1; bottom5 colCent+1];
segment8(:,:,7) = [bottom5 left5; bottom5-1 left5; bottom5 left5+1];
segment8(:,:,8) = [rowCent left5; rowCent-1 left5;rowCent+1 left5];
temp = zeros(3,8);
for k=1:8
    temp(:,k) = sub2ind([9,9],segment8(:,1,k),segment8(:,2,k));
end
EncirclementCell{2,1} = temp;

segment8 = zeros(3,2,8);
segment8(:,:,1) = [top7 left7; top7+1 left7;top7 left7+1];
segment8(:,:,2) = [top7 colCent; top7 colCent-1;top7 colCent+1];
segment8(:,:,3) = [top7 right7; top7 right7-1; top7+1 right7];
segment8(:,:,4) = [rowCent right7; rowCent-1 right7; rowCent+1 right7];
segment8(:,:,5) = [bottom7 right7; bottom7-1 right7; bottom7 right7-1];
segment8(:,:,6) = [bottom7 colCent; bottom7 colCent-1; bottom7 colCent+1];
segment8(:,:,7) = [bottom7 left7; bottom7-1 left7; bottom7 left7+1];
segment8(:,:,8) = [rowCent left7; rowCent-1 left7;rowCent+1 left7];
temp = zeros(3,8);
for k=1:8
    temp(:,k) = sub2ind([9,9],segment8(:,1,k),segment8(:,2,k));
end
EncirclementCell{3,1} = temp;

segment8 = zeros(3,2,8);
segment8(:,:,1) = [top9 left9; top9+1 left9;top9 left9+1];
segment8(:,:,2) = [top9 colCent; top9 colCent-1;top9 colCent+1];
segment8(:,:,3) = [top9 right9; top9 right9-1; top9+1 right9];
segment8(:,:,4) = [rowCent right9; rowCent-1 right9; rowCent+1 right9];
segment8(:,:,5) = [bottom9 right9; bottom9-1 right9; bottom9 right9-1];
segment8(:,:,6) = [bottom9 colCent; bottom9 colCent-1; bottom9 colCent+1];
segment8(:,:,7) = [bottom9 left9; bottom9-1 left9; bottom9 left9+1];
segment8(:,:,8) = [rowCent left9; rowCent-1 left9;rowCent+1 left9];
temp = zeros(3,8);
for k=1:8
    temp(:,k) = sub2ind([9,9],segment8(:,1,k),segment8(:,2,k));
end
EncirclementCell{4,1} = temp;

clear top3 bottom3 left3 right3 top5 bottom5 left5 right5 top7 bottom7 left7 right7;
clear top9 bottom9 left9 right9 rowCent colCent indPatch segment8;
% load dataIndianP_10Bands.mat;
 load dataPaviaU_10Bands.mat;
% [nRow,nCol,nBand] = size(im_IP10);
[nRow,nCol,nBand] = size(im_PU10);
% for k=1:nBand
%     im_IP10(:,:,k) = im_IP10(:,:,k) - min(min(im_IP10(:,:,k)));
%     im_IP10(:,:,k) = im_IP10(:,:,k)/max(max(im_IP10(:,:,k)));
% end
for k=1:nBand
    im_PU10(:,:,k) = im_PU10(:,:,k) - min(min(im_PU10(:,:,k)));
    im_PU10(:,:,k) = im_PU10(:,:,k)/max(max(im_PU10(:,:,k)));
end
% dataIndianPPatch = zeros(330,nRow*nCol);
dataPaviaUPatch = zeros(330,nRow*nCol);
kk = 0;

for k=1:nBand
%     imt = padarray(im_IP10(:,:,k),[4,4],'symmetric');
    imt = padarray(im_PU10(:,:,k),[4,4],'symmetric');
    imt = im2col(imt,[9 9],'sliding');
    ind = EncirclementCell{1,1};
%     dataIndianPPatch(kk+1:kk+9,:) = imt(ind,:);
    dataPaviaUPatch(kk+1:kk+9,:) = imt(ind,:);
    kk = kk + 9;
    for k2=2:4
        ind = EncirclementCell{k2,1};
        for k3=1:8
%             dataIndianPPatch(kk+1,:) = mean(imt(ind(:,k3),:));
            dataPaviaUPatch(kk+1,:) = mean(imt(ind(:,k3),:));
            kk = kk + 1;
        end
    end
end
% save dataIndianP_Patch.mat dataIndianPPatch;
save dataPaviaU_Patch.mat dataPaviaUPatch;
clear;pack;clc;
