clear all
close all
clc;

All_iter = 1; %set 10 independent runs

for iter = 1:All_iter
    step1_PaviaUSample;% Change start point at each iteration.
    
    %step2_LP_BS;% band selection, skip this while using the selected 10 bands:
    %dataIndianP_10Bands.mat or dataPaviaU_10Bands.mat;
    
    %step3_Generate_PatchFeatures; % patch features generation, skip this
    %while using the generated patch features: dataIndianP_Patch.mat or
    %dataPaviaU_Patch.mat;
    
    step4_PaviaU_HSP_PreTraining;% Pre-training based on the spectral information (SAE1)
    
    step5_PaviaU_SPF_PreTraining;% Pre-training based on the spatial information (SAE2)
    
    step6_PaviaU_HSP1;% fine-tuning based on the spectral information and initial labeled samples
    
    step7_PaviaU_regrow_10bands_HSP2;% region grow based on the spectral information and SAE1
    
    step8_PaviaU_SPF3;% fine-tuning based on the spatial information and grown labeled samples
    
    step9_PaviaU_regrow_10bands_SPF4;% region grow based on the spatial information and SAE2
    
    step10_PaviaU_HSP5;% fine-tuning based on the spectral information and grown labeled samples (SAE1)
    
    step11_PaviaU_SPF6;% fine-tuning based on the spatial information and grown labeled samples (SAE2)
    
    step12_PaviaU_NewProbMerge;%merge the probabilities of SAE1 and SAE2 using MRF
    
end