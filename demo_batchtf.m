clear all; clc; close all;
addpath(genpath('D:\PhD Kanan\Freelancing\Vikas Sir Paper\paper 3 idea\program\altmany-export_fig-v3.11-0-gced1834\altmany-export_fig-ced1834'))
datafolder = "D:\PhD Kanan\Freelancing\Ei\Idea 4\Dataset\Train_original\Reflux_Laryngitis";
ads = audioDatastore(datafolder, 'IncludeSubfolders',true, ...
    'FileExtensions','.wav', 'LabelSource','foldernames');
writetofolder1 = "D:\PhD Kanan\Freelancing\Ei\Idea 4\Dataset\WCOH";
helperCreatebatch_wcoh(ads,writetofolder1)

%% Batch Inage resizer
clear all; clc; close all;
imageFolder = fullfile('D:\PhD Kanan\Freelancing\Ei\Idea 4\Dataset\WCOH');
ads = imageDatastore(imageFolder,'IncludeSubfolders',true, 'LabelSource','foldernames')
while hasdata(ads)
    % Read an image.
    [B,info1] = read(ads); 
    B = imresize(B,[128 128]);
    imwrite(B,info1.Filename); 
end

