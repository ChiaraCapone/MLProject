function [t, Vm, Vs, Tm, Ts] = holdoutCVabT(X, Y, perc, nrip, intT)
% X: the dataset (test set excluded)
% Y: the labels (test set excluded)
% perc: percentage of the dataset to be used for validation
% nrip: number of repetitions of the test for each couple of parameters
% intT: list of possible iterations 
%       for example intT = [10 20 50 100 200 500 1000]
%
% Output:
% t: the value t in intT that minimize the mean of the validation error
% Vm, Vs: mean and variance of the validation error for each couple of parameters
% Tm, Ts: mean and variance of the error computed on the training set for each couple
%       of parameters
%
    
    nT = numel(intT);
 
    n = size(X,1);
    ntr = ceil(n*(1-perc));
    
    Tm = zeros(1, nT);
    Ts = zeros(1, nT);
    Vm = zeros(1, nT);
    Vs = zeros(1, nT);
    
    ym = (max(Y) + min(Y))/2;
    
    it = 0;
    for t = intT
        it = it + 1;
        for rip = 1:nrip
            I = randperm(n);
            Xtr = X(I(1:ntr),:);
            Ytr = Y(I(1:ntr),:);
            Xvl = X(I(ntr+1:end),:);
            Yvl = Y(I(ntr+1:end),:);
            
            [a, weak, ada_pred] = adaboost2(Xtr, Ytr, 10, t);
               
            trError =  calcErr(ada_pred, Ytr, ym);
            Tm(1, it) = Tm(1, it) + trError;
            Ts(1, it) = Ts(1, it) + trError^2;

            [test_pred] = adaboostTest(Xtr, Ytr, Xvl, Yvl, a, weak);
    
            valError  = calcErr(test_pred, Yvl, ym);
            Vm(1, it) = Vm(1, it) + valError;
            Vs(1, it) = Vs(1, it) + valError^2;

            str = sprintf('t\tvalErr\ttrErr\n%f\t%f\t%f\t%f\n', t, valError, trError);
            disp(str);
        end
    end
    
    Tm = Tm/nrip;
    Ts = Ts/nrip - Tm.^2;
    
    Vm = Vm/nrip;
    Vs = Vs/nrip - Vm.^2;
    
    idx = find(Vm <= min(Vm(:)));
    
    t = intT(idx(1));
end

function err = calcErr(T, Y, m)
    vT = (T >= m);
    vY = (Y >= m);
    err = sum(vT ~= vY)/numel(Y);
end
