function [perfGrid,best] = gridSearch(X,Y,R,parameterRange,state)
foldNum = 4;
trainPerc = 1-1/foldNum;
stdCoef = 1;
if state == 3
    perfGrid = zeros(foldNum,length(parameterRange),length(parameterRange),length(parameterRange));
    for fold = 1 : foldNum
        [X_tr, Y_tr, X_te, Y_te] = mtSplitPerc(X,Y,trainPerc);
        for i = 1 : length(parameterRange)
            for j = 1 : length(parameterRange)
                opts.rho_L2 = parameterRange(j);
                parfor k = 1 : length(parameterRange)
                    [W] = Least_SRMTL(X_tr,Y_tr,R,parameterRange(k),parameterRange(i),opts);
                    ypredTrain = zeros(length(Y_tr{1}),size(W,2));
                    perfTrain = zeros(1,size(W,2));
                    for t = 1 : size(W,2)
                        ypredTrain(:,t) = X_tr{t}*W(:,t);
                        [~,~,~,perfTrain(t)] = perfcurve(Y_tr{t},ypredTrain(:,t),1);
                    end
                    goodClassifiersW = find(perfTrain>(mean(perfTrain)+stdCoef*std(perfTrain)));
                    if ~isempty(goodClassifiersW)
                        ypred = zeros(length(Y_te{1}),length(goodClassifiersW));
                        for z = 1 : length(goodClassifiersW)
                            ypred(:,z) = X_te{goodClassifiersW(z)}*W(:,goodClassifiersW(z));
                            %[~,~,~,perfTest(z)] = perfcurve(Y_te{z},ypred(:,z),1);
                        end
                        [~,~,~,perfGrid(fold,i,j,k)] = perfcurve(Y_te{1},mean(ypred,2),1);
                    else
                        perfGrid(fold,i,j,k)=0;
                    end
                    disp(strcat('Grid Search Fold:',num2str(fold),',rho1=',num2str(parameterRange(i)),',rho2=',num2str(parameterRange(j)),...
                        ',rho3=',num2str(parameterRange(k)),',Perf=',num2str(perfGrid(fold,i,j,k))));
                end
                save('GridTemp3.mat','perfGrid');
            end
        end
    end
    m = squeeze(mean(perfGrid,1));
    temp = max(max(max(m)));
    [best(1),best(2),best(3)] = find(m==temp,1);
elseif state == 2
    perfGrid = zeros(foldNum,length(parameterRange),length(parameterRange));
    for fold = 1 : foldNum
        [X_tr, Y_tr, X_te, Y_te] = mtSplitPerc(X,Y,trainPerc);
        for i = 1 : length(parameterRange)
            parfor k = 1 : length(parameterRange)
                [W] = Least_SRMTL(X_tr,Y_tr,R,parameterRange(k),parameterRange(i));
                ypredTrain = zeros(length(Y_tr{1}),size(W,2));
                perfTrain = zeros(1,size(W,2));
                for t = 1 : size(W,2)
                    ypredTrain(:,t) = X_tr{t}*W(:,t);
                    [~,~,~,perfTrain(t)] = perfcurve(Y_tr{t},ypredTrain(:,t),1);
                end
                goodClassifiersW = find(perfTrain>(mean(perfTrain)+stdCoef*std(perfTrain)));
                if ~isempty(goodClassifiersW)
                    ypred = zeros(length(Y_te{1}),length(goodClassifiersW));
                    for z = 1 : length(goodClassifiersW)
                        ypred(:,z) = X_te{goodClassifiersW(z)}*W(:,goodClassifiersW(z));
                        %[~,~,~,perfTest(z)] = perfcurve(Y_te{z},ypred(:,z),1);
                    end
                    [~,~,~,perfGrid(fold,i,k)] = perfcurve(Y_te{1},mean(ypred,2),1);
                else
                    perfGrid(fold,i,k)=0;
                end
                disp(strcat('Grid Search Fold:',num2str(fold),',rho1=',num2str(parameterRange(i)),...
                    ',rho3=',num2str(parameterRange(k)),',Perf=',num2str(perfGrid(fold,i,k))));
            end
            save('GridTemp2.mat','perfGrid');
        end
    end
    m = squeeze(mean(perfGrid,1));
    temp = max(max(m));
    [best(1),best(2)] = find(m==temp,1);
elseif state == 1
    perfGrid = zeros(foldNum,length(parameterRange),length(parameterRange));
    for fold = 1 : foldNum
        [X_tr, Y_tr, X_te, Y_te] = mtSplitPerc(X,Y,trainPerc);
        for i = 1 : length(parameterRange)
            opts=cell(1,length(parameterRange));
            parfor j = 1 : length(parameterRange)
                opts{j}.rho_L2 = parameterRange(j);
                [W] = Least_Lasso(X_tr,Y_tr,parameterRange(i),opts{j});
                ypredTrain = zeros(length(Y_tr{1}),size(W,2));
                perfTrain = zeros(1,size(W,2));
                for t = 1 : size(W,2)
                    ypredTrain(:,t) = X_tr{t}*W(:,t);
                    [~,~,~,perfTrain(t)] = perfcurve(Y_tr{t},ypredTrain(:,t),1);
                end
                goodClassifiersW = find(perfTrain>(mean(perfTrain)+stdCoef*std(perfTrain)));
                if ~isempty(goodClassifiersW)
                    ypred = zeros(length(Y_te{1}),length(goodClassifiersW));
                    for z = 1 : length(goodClassifiersW)
                        ypred(:,z) = X_te{goodClassifiersW(z)}*W(:,goodClassifiersW(z));
                    end
                    [~,~,~,perfGrid(fold,i,j)] = perfcurve(Y_te{1},mean(ypred,2),1);
                else
                    perfGrid(fold,i,j)=0;
                end
                disp(strcat('Grid Search Fold:',num2str(fold),',rho1=',num2str(parameterRange(i)),',rho2=',num2str(parameterRange(j)),...
                    ',Perf=',num2str(perfGrid(fold,i,j))));
            end
            save('GridTemp1.mat','perfGrid');
        end
    end
    m = squeeze(mean(perfGrid,1));
    temp = max(max(m));
    [best(1),best(2)] = find(m==temp,1);
elseif state == 0
    perfGrid = zeros(foldNum,length(parameterRange),length(parameterRange));
    for fold = 1 : foldNum
        [X_tr, Y_tr, X_te, Y_te] = mtSplitPerc(X,Y,trainPerc);
        for i = 1 : length(parameterRange)
            opts=cell(1,length(parameterRange));
            parfor j = 1 : length(parameterRange)
                opts{j}.rho_L2 = parameterRange(j);
                [W] = Least_Lasso(X_tr,Y_tr,parameterRange(i),opts{j});
                ypred = X_te{1}*W;
                [~,~,~,perfGrid(fold,i,j)] = perfcurve(Y_te{1},ypred,1);
                disp(strcat('Grid Search Fold:',num2str(fold),',rho1=',num2str(parameterRange(i)),',rho2=',num2str(parameterRange(j)),...
                    ',Perf=',num2str(perfGrid(fold,i,j))));
            end
            save('GridTemp1.mat','perfGrid');
        end
    end
    m = squeeze(mean(perfGrid,1));
    temp = max(max(m));
    [best(1),best(2)] = find(m==temp,1);
end

