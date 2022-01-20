clear; clc; close all;

% Load the directories
datafolder_type1 = "D:\PhD Kanan\Freelancing\Ei\Idea 4\Result\Generated Images64x64\OS8";
allImages_type1 = imageDatastore(datafolder_type1,'IncludeSubfolders',true,'LabelSource','foldernames');
[imgsTrain_type1,imgsTest_type1] = splitEachLabel(allImages_type1,0.75,'randomized');
disp(['Number of training images: ',num2str(numel(imgsTrain_type1.Files))]);
countEachLabel(imgsTrain_type1)
disp(['Number of test images: ',num2str(numel(imgsTest_type1.Files))]);
countEachLabel(imgsTest_type1)

augmenter = imageDataAugmenter('RandXReflection',true);
augimds_train = augmentedImageDatastore([64 64 3],imgsTrain_type1,'DataAugmentation',augmenter,'ColorPreprocessing','gray2rgb');
augimds_test = augmentedImageDatastore([64 64 3],imgsTest_type1,'DataAugmentation',augmenter,'ColorPreprocessing','gray2rgb');

% Define the layers

% layers = [
%     imageInputLayer([64 64 3],"Name","imageinput")
%     convolution2dLayer([3 3],8,"Name","conv_1","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_1")
%     reluLayer("Name","relu_1")
%     convolution2dLayer([3 3],16,"Name","conv_2","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_2")
%     reluLayer("Name","relu_2")
%     convolution2dLayer([3 3],32,"Name","conv_3","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_3")
%     reluLayer("Name","relu_3")
%     convolution2dLayer([3 3],64,"Name","conv_4","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_4")
%     reluLayer("Name","relu_4")
%     convolution2dLayer([3 3],128,"Name","conv_5","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_5")
%     reluLayer("Name","relu_5")
%     convolution2dLayer([3 3],256,"Name","conv_6","Padding","same")
%     dropoutLayer(0.5,"Name","dropout")
%     reluLayer("Name","relu_6")
%     fullyConnectedLayer(4,"Name","fc")
%     softmaxLayer("Name","softmax")
%     classificationLayer("Name","classoutput")];

layers = [
    imageInputLayer([64 64 3],"Name","imageinput")
    convolution2dLayer([3 3],8,"Name","conv_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer([3 3],16,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer([3 3],32,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer([3 3],64,"Name","conv_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_4")
    convolution2dLayer([3 3],128,"Name","conv_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_5")
    convolution2dLayer([3 3],256,"Name","conv_6","Padding","same")
    dropoutLayer(0.5,"Name","dropout")
    reluLayer("Name","relu_6")
    fullyConnectedLayer(4,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

miniBatchSize = 64;
valFrequency = 30;
options_type1 = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',400, ...
    'InitialLearnRate',0.00001, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimds_test, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

trained_net = trainNetwork(augimds_train,layers,options_type1);
%%
clc
[YPred,probs] = classify(trained_net,augimds_test);
accuracy = mean(YPred == imgsTest_type1.Labels);
classes = trained_net.Layers(end).Classes;
cnnResponses = predict(trained_net,augimds_test);
[~,classIdx] = max(cnnResponses,[],2);
predictedLabels = classes(classIdx);
figure
cm = confusionchart(imgsTest_type1.Labels,predictedLabels);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
% fprintf('Average accuracy of googlenet_amor model = %0.2f\n',accuracy_googlenet*100)
C = confusionmat(imgsTest_type1.Labels,predictedLabels); % This is the confusion matrix
[Result,RefereceResult]=confusion.getValues(C);
disp(Result)


