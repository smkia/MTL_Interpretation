function [X_sel, Y_sel, X_res, Y_res, selIdx] = splitData(X, Y, percent)
task_num = size(X);
selIdx = cell(task_num, 0);
X_sel = cell(task_num, 0);
Y_sel = cell(task_num, 0);
X_res = cell(task_num, 0);
Y_res = cell(task_num, 0);
task_sample_size = length(Y{1});
tSelIdx = randperm(task_sample_size) <task_sample_size * percent;
for t = 1: task_num
    selIdx{t} = tSelIdx;
    X_sel{t} = X{t}(tSelIdx,:);
    Y_sel{t} = Y{t}(tSelIdx,:);
    X_res{t} = X{t}(~tSelIdx,:);
    Y_res{t} = Y{t}(~tSelIdx,:);
    
end