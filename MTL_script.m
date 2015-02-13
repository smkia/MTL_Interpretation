%% To run this script you need first to download MALSAR toolbox, and to add its path to your MATLAB search paths.

%% Parameters
clear all;
subjects_train = 1:5;
foldNum = 10;
parameterRange = [0 0.001 0.01 0.1 1 10 50 100];
timeInterval = 76:325; % -200ms to 800ms
fileAddress = '';

 %% Single Subject - Multi task - Elastic Net
for subj = 1 : length(subjects_train)
    % Preparing data
    filename = sprintf(strcat(fileAddress,'/train_subject%02d.mat'),subjects_train(subj));
    disp(strcat('Loading ',filename));
    data = load(filename);
    [trialNum,channelNum,timeNum] = size(data.X);
    data.y(data.y==0)=-1;
    data.X = data.X(:,:,timeInterval);
    for i = 1 : channelNum
        d{i} = squeeze(data.X(:,i,:));
        d{i} = mapstd(d{i}');
        d{i} = d{i}';
        target{i} = single(data.y);
    end
    clear data;
    % Training
    training_percent = 0.9;
    for f = 1 : foldNum
        [X_tr, Y_tr, X_te, Y_te] = splitData(d,target,training_percent);
        if f == 1
            [perfGrid{subj},bestParam{subj}] = gridSearch(X_tr,Y_tr,[],parameterRange,1);
        end
        opts.rho_L2 = parameterRange(bestParam{subj}(2));
        [W{f,subj},C{f,subj}] = Least_Lasso(X_tr,Y_tr,parameterRange(bestParam{subj}(1)),opts);
        ypredTrain = zeros(length(Y_tr{1}),size(W{f,subj},2));
        for t = 1 : size(W{f,subj},2)
            ypredTrain(:,t) = X_tr{t}*W{f,subj}(:,t);
            [~,~,~,perfTrain{f,subj}(t)] = perfcurve(Y_tr{t},ypredTrain(:,t),1);
        end
        goodClassifiersW{f,subj} = find(perfTrain{f,subj}>(mean(perfTrain{f,subj})+1*std(perfTrain{f,subj})));
        ypred = [];
        for i = 1 : length(goodClassifiersW{f,subj})
            ypred(:,i) = X_te{goodClassifiersW{f,subj}(i)}*W{f,subj}(:,goodClassifiersW{f,subj}(i));
            [~,~,~,perfTest{f,subj}(i)] = perfcurve(Y_te{goodClassifiersW{f,subj}(i)},ypred(:,i),1);
        end
        [~,~,~,perfTotal(f,subj)] = perfcurve(Y_te{1},mean(ypred,2),1);
        correctedW{f,subj} = zeros(size(W{f,subj}));
        correctedW{f,subj}(:,goodClassifiersW{f,subj}) = W{f,subj}(:,goodClassifiersW{f,subj});
        for i = 1 : 306
            timeCorrectedW{f,subj}(:,i) = cov(d{i})*W{f,subj}(:,i);
        end
        disp(strcat('Subject:',num2str(subj),',Fold:',num2str(f),',GCN:',num2str(length(goodClassifiersW{f,subj})),',AUC:',num2str(perfTotal(f,subj))));
    end
    clear 'd' 'target';
    save('Results_ENMTL.mat','W','C','perfTrain','goodClassifiersW','perfTest','perfTotal','correctedW','timeCorrectedW','perfGrid','bestParam');
end

%% Single Subject - Single task

for subj = 1 : length(subjects_train)
    % Preparing data
    filename = sprintf(strcat(fileAddress,'/train_subject%02d.mat'),subjects_train(subj));
    disp(strcat('Loading ',filename));
    data = load(filename);
    [trialNum,channelNum,timeNum] = size(data.X);
    data.y(data.y==0)=-1;
    data.X = data.X(:,:,timeInterval);
    d = [];
    d{1} = reshape(data.X,trialNum,channelNum*length(timeInterval));
    d{1} = mapstd(d{1}');
    d{1} = d{1}';
    target{1} = single(data.y);
    clear data;
    % Training
    training_percent = 0.9;
    for f = 1 : foldNum
        [X_tr, Y_tr, X_te, Y_te] = splitData(d,target,training_percent);
        if f == 1
            [perfGrid{subj},bestParam{subj}] = gridSearch(X_tr,Y_tr,[],parameterRange,0);
        end
        opts.rho_L2 = parameterRange(bestParam{subj}(2));
        [W{f,subj},C{f,subj}] = Least_Lasso(X_tr,Y_tr,parameterRange(bestParam{subj}(1)),opts);
        ypredTrain = X_tr{1}*W{f,subj};
        [~,~,~,perfTrain(f,subj)] = perfcurve(Y_tr{1},ypredTrain,1);
        ypred = X_te{1}*W{f,subj};
        [~,~,~,perfTest(f,subj)] = perfcurve(Y_te{1},ypred,1);
        timeCorrectedW{f,subj} = inpCov*W{f,subj};
        disp(strcat('Subject:',num2str(subj),',Fold:',num2str(f),',AUC:',num2str(perfTest(f,subj))));
    end
    clear 'd' 'target' 'inpCov';
    save('Results_LassoST.mat','W','C','perfTrain','perfTest','perfGrid','bestParam'); %,'timeCorrectedW'
end

%% Single Subject - channel wise single task 

for subj = 1 : length(subjects_train)
    % Preparing data
    filename = sprintf(strcat(fileAddress,'/train_subject%02d.mat'),subjects_train(subj));
    disp(strcat('Loading ',filename));
    data = load(filename);
    [trialNum,channelNum,timeNum] = size(data.X);
    data.y(data.y==0)=-1;
    data.X = data.X(:,:,timeInterval);
    for i = 1 : channelNum
        d{i} = squeeze(data.X(:,i,:));
        d{i} = mapstd(d{i}');
        d{i} = d{i}';
        target{i} = single(data.y);
    end
    clear data;
    % Training
    training_percent = 0.9;
    for f = 1 : foldNum
        [X_tr, Y_tr, X_te, Y_te] = splitData(d,target,training_percent);
        for ch = 1 : channelNum
            XC_tr{1} = X_tr{ch};
            YC_tr{1} = Y_tr{ch};
            if f == 1
                [perfGrid{subj,ch},bestParam{subj,ch}] = gridSearch(XC_tr,YC_tr,[],parameterRange,0);
            end
            opts.rho_L2 = parameterRange(bestParam{subj,ch}(2));
            [W{f,subj}(:,ch)] = Least_Lasso(XC_tr,YC_tr,parameterRange(bestParam{subj,ch}(1)),opts);
        end
        ypredTrain = zeros(length(Y_tr{1}),size(W{f,subj},2));
        for t = 1 : size(W{f,subj},2)
            ypredTrain(:,t) = X_tr{t}*W{f,subj}(:,t);
            [~,~,~,perfTrain{f,subj}(t)] = perfcurve(Y_tr{t},ypredTrain(:,t),1);
        end
        goodClassifiersW{f,subj} = find(perfTrain{f,subj}>(mean(perfTrain{f,subj})+1*std(perfTrain{f,subj})));
        ypred = [];
        for i = 1 : length(goodClassifiersW{f,subj})
            ypred(:,i) = X_te{goodClassifiersW{f,subj}(i)}*W{f,subj}(:,goodClassifiersW{f,subj}(i));
            [~,~,~,perfTest{f,subj}(i)] = perfcurve(Y_te{goodClassifiersW{f,subj}(i)},ypred(:,i),1);
        end
        [~,~,~,perfTotal(f,subj)] = perfcurve(Y_te{1},mean(ypred,2),1);
        correctedW{f,subj} = zeros(size(W{f,subj}));
        correctedW{f,subj}(:,goodClassifiersW{f,subj}) = W{f,subj}(:,goodClassifiersW{f,subj});
        for i = 1 : 306
            timeCorrectedW{f,subj}(:,i) = cov(d{i})*W{f,subj}(:,i);
        end
        disp(strcat('Subject:',num2str(subj),',Fold:',num2str(f),',GCN:',num2str(length(goodClassifiersW{f,subj})),',AUC:',num2str(perfTotal(f,subj))));
    end
    clear 'd' 'target';
    save('Results_LassoCHST.mat','W','perfTrain','goodClassifiersW','perfTest','perfTotal','correctedW','timeCorrectedW','perfGrid','bestParam');
end

