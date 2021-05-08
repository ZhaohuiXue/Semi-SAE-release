clear all
close all
clc;

All_iter = 1; %set 10 independent runs

for iter = 1:All_iter
    step1_IndianPSample;% Change start point at each iteration.
    
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
    
end