%
% Semi-SAE: Semisupervised stacked autoencoders for hyperspectral image classification DEMO.
%        Version: 1.0
%        Date   : May 2021
%
%    This demo shows the Semi-SAE method for hyperspectral image classification.
%
% /data   ... The folder contains the original hyperspectral data, the selected 10 bands data, the patch features for the IP and PU % data sets.
% /LP-BS ... The folder contains LB-PS code for band selection.
% /run_IP_codes ... The folder contains the main codes for IP.
% /run_PU_codes ... The folder contains the main codes for PU.
% /UFLDL-Tutorial-Exercise-master ... The folder contains the tutorial exercise by Andrew Ng.
% regionadjacency.m ... The function used to computes adjacency matrix for image of labeled segmented regions.
% regiongrowHHUZhou.m ... The function used for regiongrow in a multispectral image.


% Main steps (take IP for example):
    step1_IndianPSample;% Select the initial training and test samples.
    
    %step2_LP_BS;% band selection, skip this while using the selected 10 bands:
    %dataIndianP_10Bands.mat or dataPaviaU_10Bands.mat;
    
    %step3_Generate_PatchFeatures; % patch features generation, skip this
    %while using the generated patch features: dataIndianP_Patch.mat or
    %dataPaviaU_Patch.mat;
    
    step4_IndianP_HSP_PreTraining;% Pre-training based on the spectral information (SAE1)
    
    step5_IndianP_SPF_PreTraining;% Pre-training based on the spatial information (SAE2)
    
    step6_IndianP_HSP1;% fine-tuning based on the spectral information and initial labeled samples
    
    step7_IndianP_regrow_10bands_HSP2;% region grow based on the spectral information and SAE1
    
    step8_IndianP_SPF3;% fine-tuning based on the spatial information and grown labeled samples
    
    step9_IndianP_regrow_10bands_SPF4;% region grow based on the spatial information and SAE2
    
    step10_IndianP_HSP5;% fine-tuning based on the spectral information and grown labeled samples (SAE1)
    
    step11_IndianP_SPF6;% fine-tuning based on the spatial information and grown labeled samples (SAE2)
    
    step12_IndianP_NewProbMerge;%merge the probabilities of SAE1 and SAE2 using MRF
%
%   --------------------------------------
%   Note: Required toolbox/functions are covered
%   --------------------------------------
%  1. LP-BS: Provided by Jenny Du.
%  2. Peter Kovesi_seg_functions: https://www.peterkovesi.com/matlabfns/index.html#segmentation (for regionadjacency.m).
%  3. UFLDL-Tutorial-Exercise-master: https://github.com/dkyang/UFLDL-Tutorial-Exercise.
%   --------------------------------------
%   Cite:
%   --------------------------------------
%
%   [1]S. G. Zhou, Z. H. Xue, P. J. Du. Semisupervised Stacked Autoencoder With Cotraining for Hyperspectral Image Classification[J]. IEEE Transactions on Geoscience and Remote Sensing, 2019, 57(6): 3813-3826.
%
%   --------------------------------------
%   Copyright & Disclaimer
%   --------------------------------------
%
%   The programs contained in this package are granted free of charge for
%   research and education purposes only. 
%
%   Copyright (c) 2021 by Zhaohui Xue
%   zhaohui.xue@hhu.edu.cn
%   --------------------------------------
%   For full package:
%   --------------------------------------
%   https://sites.google.com/site/zhaohuixuers/




