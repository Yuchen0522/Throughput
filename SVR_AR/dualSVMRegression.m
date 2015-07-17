function [params_linear, svs_linear] = dualSVMRegression(K, y, c, eps)
n=length(y);
one=ones(n,1);
cvx_begin
    variable a1(n) nonnegative;
    variable a2(n) nonnegative;
    maximize( -0.5*(a1-a2)'*K*(a1-a2)-eps*(one'*(a1+a2))+y'*(a1-a2) );
    subject to
        a1 <= c;
        a2 <= c;
cvx_end

a=a1-a2;
zero_indices=find(abs(a)<0.0001);
svs_linear=ones(n,1);
svs_linear(zero_indices)=0;
a(zero_indices)=0;
params_linear=a;