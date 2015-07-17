function [preds_linear] = dualSVMPredict(params_linear, K)
preds_linear=sum(bsxfun(@times,params_linear,K))';

