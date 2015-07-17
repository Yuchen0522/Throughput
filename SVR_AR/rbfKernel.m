function [K] = rbfKernel(X,Y,h)
K=zeros(length(X),length(Y));
for i=1:length(X)
    for j=1:length(Y)
        K(i,j)=exp(-((X(i,:)-Y(j,:))*(X(i,:)-Y(j,:))')/(2*h));
    end
end

            
            
