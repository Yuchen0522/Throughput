function []=regression(city)
%part1 read in data
load('yShanghai');
load('yShenzhen');
if strcmp(city,'shanghai')==0  
    series=yShanghai;
end
if strcmp(city,'shenzhen')==0  
    series=yShenzhen;
end

%plot data
figure;
plot(yShanghai,'r');hold on;
plot(yShenzhen,'b');
title('Throughput from 2001/01-2014/10');
xlabel('Month');
ylabel('Throughput');
legend('Shanghai','Shenzhen');


%part2 transfer and partition
n=length(series);
p=12; %tune, autoregressive order
data=transfer(series,p);
nTotal=n-p;
nTest=11; %tune, number of data points to test, must less than nTotal, which is n-p
nTrain=nTotal-nTest;
xTrain=data(1:nTrain,1:p);
yTrain=data(1:nTrain,p+1);
xTest=data(nTrain+1:nTotal,1:p);
yTest=data(nTrain+1:nTotal,p+1);


%part3 SV Regression
eps = 10;%tune
c = 400;%tune

% Parameters for the Kernel
h = std(xTrain)*std(xTrain)';%tune


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct the rbf Kernel here.
Krbf = rbfKernel(xTrain,xTrain,h);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solve the Dual problem
% You should return any information you will need to obtain predictions in the
% struct params and the support vectors in svs. svs should be a matrix with 1
% if a point is a support vector and 0 otherwise.
[params_linear, svs_linear] = dualSVMRegression(xTrain*xTrain', yTrain, c, eps);
[params_rbf, svs_rbf] = dualSVMRegression(Krbf, yTrain, c, eps); 

% Obtain the predictions
yPredictedLinear = dualSVMPredict(params_linear, xTrain*[xTrain;xTest]');
yPredictedRbf = dualSVMPredict(params_rbf, rbfKernel(xTrain, [xTrain;xTest], h));
% calculate error
%Abs
eTrainLinearAbs=mean(abs(yPredictedLinear(1:nTrain)-yTrain));
eTrainRbfAbs=mean(abs(yPredictedRbf(1:nTrain)-yTrain));
eTestLinearAbs=mean(abs(yPredictedLinear(nTrain+1:nTotal)-yTest));
eTestRbfAbs=mean(abs(yPredictedRbf(nTrain+1:nTotal)-yTest));
%Rmse
eTrainLinearRmse=norm(yPredictedLinear(1:nTrain)-yTrain);
eTrainRbfRmse=norm(yPredictedRbf(1:nTrain)-yTrain);
eTestLinearRmse=norm(yPredictedLinear(nTrain+1:nTotal)-yTest);
eTestRbfRmse=norm(yPredictedRbf(nTrain+1:nTotal)-yTest);


%part4 Draw figure
% Now plot the estimate and the support vectors
t = linspace(1,nTotal,nTotal)'; % time
figure;
plot( t(1:nTrain), yTrain, 'bo'); hold on,
plot( t(nTrain+1:nTotal), yTest, 'rx'); 
plot( t, yPredictedLinear, 'b');
plot( t, yPredictedRbf, 'g');
xlabel('Months');
ylabel('Throughput')
title('Linear/Rbf Kernel');
legend('Train','Test','Linear','Rbf');
fprintf('Linear Abs Error: Train %0.4f, Test %0.4f\n', eTrainLinearAbs, eTestLinearAbs);
fprintf('Linear Rmse Error: Train %0.4f, Test %0.4f\n', eTrainLinearRmse, eTestLinearRmse);
fprintf('Rbf Abs Error: Train %0.4f, Test %0.4f\n', eTrainRbfAbs, eTestRbfAbs);
fprintf('Rbf Rmse Error: Train %0.4f, Test %0.4f\n', eTrainRbfRmse, eTestRbfRmse);

