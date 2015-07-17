function [data]=transfer(series,p)
n=length(series);
data=zeros(n-p,p+1);%allocate matrix, first p columns are xTrain/xTest, last column is yTrain/yTest
for i=1:p+1
    data(:,i)=series(i:n-p+i-1);
end
