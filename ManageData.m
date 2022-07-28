function [TrainData,TestData] = ManageData(data)

Labels = data(:,end);
Features = data(:,1:end-1);

%Features = Normal(Features);


% Divide TrianData And TestData
pT = 75;
[nSamples,nFeature] = size(Features);
nTrain = round((pT/100)*nSamples);

R = randperm(nSamples);
indTrain = R(1:nTrain);
indTest = R(nTrain+1:end);

TrainData.Feature = Features(indTrain,:);
TrainData.Lebel = Labels(indTrain);
TrainData.nFeature = nFeature;

TestData.Feature = Features(indTest,:);
TestData.Lebel = Labels(indTest);


% % Apply K-Fold Cross Validation
% K = 4;
% NT = numel(indTrain);
% CVI = crossvalind('kfold',NT,K);
% TrainData.CVI = CVI;
% TrainData.K = K;

end