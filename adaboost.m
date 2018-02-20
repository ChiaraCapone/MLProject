function [a, weak, ada_pred] = adaboost(Xtr, Ytr, algo, n, T)
% output to use for cross validation: [a, weak, ada_pred]
% [err_new, a, W, R, H, H_f, weak, pred]
    
    % n = number of weak learners
    % T = number of iterations
    
    nLambda = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 1.5, 5, 10, 15];
    sizeX = size(Xtr,1);
    W = zeros(sizeX,T+1);
    h = zeros(sizeX,n);
    a = zeros(1,T);
    err_new = zeros(T,n);
    R = ones(3,T);
    min_err = zeros(T,T);
    H = zeros(sizeX,T);
    H_f = ones(1,T);
    weak = {};
    pred = zeros(sizeX, T);
    weak = []; % array of the indices of the lambda
    
    % fill matrix W with the uniform distribution (weight matrix)
    for i = 1:sizeX
        W(i,1) = 1/sizeX;
    end
          
    for t = 1:T
        
        if isequal(algo, 'tree')
            %tree = TreeBagger(n, Xtr, Ytr, 'Weights', W(:,t), 'MaxNumSplits', 1, 'MergeLeaves', 'off', 'Prune', 'off');
            tree = TreeBagger(n, Xtr, Ytr, 'Weights', W(:,t), 'MergeLeaves', 'off', 'Prune', 'off');
            for i = 1:n
                h(:,i) = str2double(predict(tree.Trees{1,i}, Xtr));
                for j = 1:sizeX
                    if h(j,i) ~= Ytr(j)
                        err_new(t,i) = err_new(t,i) + W(j,t);
                    end
                end
            end
        end
        if isequal(algo, 'rls')
            n = 10;
            h = callRls(Xtr, Ytr, h, nLambda);
            for i = 1:n
                for j = 1:sizeX
                    if h(j,i) ~= Ytr(j)
                        err_new(t,i) = err_new(t,i) + W(j,t);
                    end
                end
            end
        end
        
        min_err(t,t) = min(err_new(t,:));
        if min_err(t,t) == 1/2
            disp('error = 0.5');
            break;
        end
        
        index_min = find(err_new(t,:) == min_err(t,t), 1, 'first');
        
        if isequal(algo, 'tree')
            weak{t} = tree.Trees{1,index_min};
        end
        if isequal(algo, 'rls')
            pred(:,t) = h(:,index_min);
            weak(:,t) = nLambda(1,index_min);
        end
        
        if min_err(t,t) > 1/2
            disp('error > 0.5');
            a(1,t) = 0;
            break;
        else
            a(1,t) = 0.5*log((1-min_err(t,t))/min_err(t,t));
        end
        
        % modify using index at line 45         
        R(1,t) = index_min;
        R(2,t) = err_new(t,index_min);
        R(3,t) = a(t);
                
        %Z = 2*sqrt(min_err(t,t)*(1-min_err(t,t)));
        
        % update weight matrix W
        for i = 1:sizeX
            if h(i,R(1,t)) == Ytr(i)
                %W(i,t+1) = (1/Z) * W(i,t)*exp(-a(t));
                W(i,t+1) = 0.5*(1/(1-min_err(t,t)))*W(i,t);
                %W(i,t+1) = W(i,t)*exp(-a(t));
            else
                %W(i,t+1) = (1/Z) * W(i,t)*exp(a(t));
                W(i,t+1) = 0.5*(1/min_err(t,t))*W(i,t);
                %W(i,t+1) = W(i,t)*exp(a(t));
            end
        end       
    end
    
    if isequal(algo, 'tree')
        for i = 1:t
            pred(:,i) = str2double(weak{i}.predict(Xtr));
        end
    end
    
    i = 1;
    H_f(1,i) = calcErr(H(:,i),Ytr);

    while i<t
        for j = 1:i
            H(:,i) = H(:,i) + a(1,j)*pred(:,j);
            H_f(1,i) = calcErr(H(:,i),Ytr);
            ada_train = H_f(1,i);
        end
        i = i+1;
    end
    disp(i);
    for j = 1:i
        H(:,t) = H(:,t) + a(1,j)*pred(:,j);
    end
    
    H_f(1,t) = calcErr(H(:,t),Ytr);
    H = sign(H);
    ada_pred = H(:,end);
    %ada_tr = calcErr(ada_pred,Ytr);
end

function h = callRls(Xtr, Ytr, h, nLambda)
    
    i = 1;
    
    for l = nLambda
        w = regularizedLSTrain(Xtr, Ytr, l);
        h(:,i) = regularizedLSTest(w, Xtr);
        i = i+1;
    end
end