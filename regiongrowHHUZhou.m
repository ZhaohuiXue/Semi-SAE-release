function [ind] = regiongrowHHUZhou(im,seed )
% The function used for regiongrow in a  multispectral image.
%  F is the input multispectral image.
% seed is the coordinates of the seed pixel.
f = im;
[nRow, nCol, nSli] = size(f);
maxW = 7;
factor1 = 0.5; 
factor2 = floor(nSli/3);  %
% factor2 = 2;%0.5/2
seedVal = squeeze(f(seed(1),seed(2),:));
% threshVal = zeros(nSli,1);
% for k=1:nSli
%     maxV = max(max(f(:,:,k)));
%     minV = min(min(f(:,:,k)));
%     threshVal(k) = (maxV - minV) * 0.2;
% end

rowUP = max(1,seed(1)-maxW);
rowBOTTOM = min(nRow, seed(1)+maxW);

colLEFT = max(1,seed(2)-maxW);
colRIGHT = min(nCol, seed(2)+maxW);

temPATCH = f(rowUP:rowBOTTOM,colLEFT:colRIGHT,:);
mPATCH = rowBOTTOM - rowUP + 1;
nPATCH = colRIGHT - colLEFT + 1;

if seed(1)-maxW<1
    seed_row_in_patch = seed(1);
else
    seed_row_in_patch = maxW + 1;
end

if seed(2)-maxW<1
    seed_col_in_patch = seed(2);
else
    seed_col_in_patch = maxW + 1;
end

BW = true(mPATCH,nPATCH);

bandDIFF = zeros(numel(temPATCH(:,:,1)),nSli);
for k=1:nSli
    tempDIFF = abs(temPATCH(:,:,k) - seedVal(k));
    [~,ind] = sort(tempDIFF(:));
    selectedN = round(length(ind)*factor1);
    bandDIFF(ind(1:selectedN),k) = 1; 
end
ind = sum(bandDIFF,2)>nSli-factor2;
BW = reshape(ind,mPATCH,nPATCH);
BW2 = bwselect(BW, seed_col_in_patch,seed_row_in_patch,4);
[rt,ct] = find(BW2);
grownPATCH = [rt - seed_row_in_patch + seed(1) , ct - seed_col_in_patch + seed(2)];
% BW = false(nRow, nCol);
ind = sub2ind([nRow, nCol], grownPATCH(:,1),grownPATCH(:,2));
% BW(ind) = true;
end

