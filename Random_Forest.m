clc
clear
close all; 

%Loading
load data1.csv
[TrainData,TestData] = ManageData(data1);
train_x = TrainData.Feature;
target = TrainData.Lebel;
test_x = TestData.Feature;
test_y = TestData.Lebel ;
data1 = train_x(:,1);
data2 = train_x(:,2);
data3 = train_x(:,3);
data4 = train_x(:,4);
data5 = train_x(:,5);

X = table(data1,data2,data3,data4,data5,target);
rng('default'); % For reproducibility

%%Specify Tuning Parameters
maxMinLS = 50;
minLS = optimizableVariable('minLS',[1,maxMinLS],'Type','integer');
numPTS = optimizableVariable('numPTS',[1,size(X,2)-1],'Type','integer');
hyperparametersRF = [minLS; numPTS];

%Minimize Objective Using Bayesian Optimization
results = bayesopt(@(params)oobErrRF(params,X,target),hyperparametersRF,...
    'AcquisitionFunctionName','expected-improvement-plus','Verbose',0);

bestOOBErr = results.MinObjective
bestHyperparameters = results.XAtMinObjective

%Train Model Using Optimized Hyperparameters
Mdl = TreeBagger(450,train_x,target,'OOBPred','On','Method','regression',...
    'MinLeafSize',bestHyperparameters.minLS,...
    'NumPredictorstoSample',bestHyperparameters.numPTS);


predicted_train = oobPredict(Mdl);
predicted_test = predict(Mdl,test_x);
trainmse = sum((predicted_train-target).^2)/length(target);
testmse = sum((predicted_test-test_y).^2)/length(test_y);

[fitresult.train, gof.train] = fit( predicted_train, target, 'poly1' );
[fitresult.test, gof.test] = fit( predicted_test, test_y, 'poly1' );
b = gof.train.rsquare;
c = gof.test.rsquare;

figure
plot(fitresult.train,predicted_train,target)
hold on
title(['R-Square = ' num2str(b)])

figure
plot(fitresult.test,predicted_test,test_y)
hold on
title(['R-Square = ' num2str(c)])

figure
plot(predicted_train,':og')
hold on
plot(target,'- *')
title('Train')

figure
plot(predicted_test,':og')
hold on
plot(test_y,'- *')
title('Test')

MAPE.train = mean((abs(predicted_train-target))./target).*100;
MAPE.test = mean((abs(predicted_test-test_y))./test_y).*100;

Vmse.train=errperf(target,predicted_train,'mse');
Vmse.test=errperf(test_y,predicted_test,'mse');
RMSE.train=sqrt(Vmse.train);
RMSE.test=sqrt(Vmse.test);
