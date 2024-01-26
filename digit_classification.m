digitDatasetPath = 'C:\Users\RAMA\OneDrive\Documents\MATLAB\DigitDataset';
digitimages = imageDatastore(digitDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
numTrainFiles = 750;
[TrainImages,TestImages]=splitEachLabel(digitimages,numTrainFiles,'randomize');
layers=[ imageInputLayer([28 28 1],'Name','Input')
          convolution2dLayer(3,8,'Padding','same','Name','Conv_1')
          batchNormalizationLayer('Name','BN_1')
          reluLayer('Name','Relu_1')
          maxPooling2dLayer(2,'Stride',2,'Name','MaxPool_1')

          convolution2dLayer(3,16,'Padding','same','Name','Conv_2')
          batchNormalizationLayer('Name','BN_2')
          reluLayer('Name','Relu_2')
          maxPooling2dLayer(2,'Stride',2,'Name','MaxPool_2')
 
          convolution2dLayer(3,32,'Padding','same','Name','Conv_3')
          batchNormalizationLayer('Name','BN_3')
          reluLayer('Name','Relu_3')
          maxPooling2dLayer(2,'Stride',2,'Name','MaxPool_3')
          
          convolution2dLayer(3,64,'Padding','same','Name','Conv_4')
          batchNormalizationLayer('Name','BN_4')
          reluLayer('Name','Relu_4')

          fullyConnectedLayer(10,'Name','FC');
          softmaxLayer('Name','SoftMax');
          classificationLayer('Name','Output Classification');
          ];
           
options = trainingOptions('sgdm','MaxEpochs', 5,'MiniBatchSize', 128,'InitialLearnRate', 0.01,'Shuffle', 'every-epoch','ValidationData',TestImages,'ValidationFrequency',30,'Verbose', false,'Plots', 'training-progress');
net = trainNetwork(TrainImages,layers, options);

% Evaluate the performance
YTestPred = classify(net,TestImages);
YTest=TestImages.Labels;
accuracy = sum(YTestPred == YTest) / numel(YTest);
confMat = confusionmat(YTest, YTestPred);
precision = diag(confMat) ./ sum(confMat, 2);
recall = diag(confMat) ./ sum(confMat, 1)';
f1Score = 2 * (precision .* recall) ./ (precision + recall);

disp('Accuracy:');
disp(accuracy);
disp('Confusion Matrix:');
disp(confMat);
disp('Precision:');
disp(precision');
disp('Recall:');
disp(recall');
disp('F1 Score:');
disp(f1Score');