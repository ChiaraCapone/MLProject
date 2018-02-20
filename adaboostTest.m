function [test_pred] = adaboostTest(Xtr, Ytr, Xts, Yts, a, algo, weak)
    % output to use for cross validation: [test_pred]
    % output for drawing graphs: [Ypred, Hf_test, err_init, ada_test]
    
    % a = array of weak learners weights found during training
    % algo = string representing the type of weak classifiers to use
    % weak = set of weak classifiers chosen during training
    
    sizeX = size(Xts,1);
    n = size(weak,2);
    disp(n);
    h = zeros(sizeX,n);
    Ypred = zeros(sizeX,1);
    H = zeros(sizeX,n);
    Hf_test = ones(1,n);
    err_init = zeros(1,n);
        
    if isequal(algo, 'tree')
        for i = 1:n
            h(:,i) = str2double(predict(weak{i}, Xts));
            err_init(1,i) = calcErr(h(:,i), Yts);
        end
    end
    
    if isequal(algo, 'rls')
        h = callRls(Xtr, Ytr, Xts, h, weak);
        for i = 1:n
            err_init(1,i) = calcErr(h(:,i), Yts);
        end
    end
    
    for i = 1:n
        Ypred(:) = Ypred(:) + a(1,i)*h(:,i);
    end
    
    Ypred = sign(Ypred);
    
    for i = 1:n
        for j = 1:i
           H(:,i) = H(:,i) + a(1,j)*h(:,j);
           Hf_test(1,i) = calcErr(H(:,i),Yts);
        
        end
    end
    %test_pred = H(:,T);
    ada_test = calcErr(Ypred, Yts);
    end

    
function h = callRls(Xtr, Ytr, Xts, h, nLambda)

    i = 1;
    
    for l = nLambda
        w = regularizedLSTrain(Xtr, Ytr, l);
        h(:,i) = regularizedLSTest(w, Xts);
        i = i+1;
    end
end