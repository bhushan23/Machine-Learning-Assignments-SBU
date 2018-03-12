
X = trD;
Y = double(trLb);
C = 10;
  

[WTrain, BTrain, TObj] = train(X, Y, C);
[trainAcc, trainConf, trainSV, TrainObj] = predictFun(X, Y, WTrain, BTrain, C)

X = valD;
Y = valLb;
%C = 0.1;
%WTrain, BTrain = train(X, Y, C);
[valAcc, valConf, valSV, valObj] = predictFun(X, Y, WTrain, BTrain, C)

% 1 Accuracy
ToPrint = ['C: ', num2str(C), ' Train Accuracy: ', num2str(trainAcc), ' Validation Accuracy: ', num2str(valAcc)];
disp(ToPrint)

ToPrint = ['C: ', num2str(C), ' Train SV: ', num2str(trainSV), ' Validation SV: ', num2str(valSV)];
disp(ToPrint)

ToPrint = ['C: ', num2str(C), ' Train Obj: ', num2str(trainObj), ' Validation Obj: ', num2str(valObj)];
disp(ToPrint)

disp(num2str(TObj))
% 2 Objective value of SVM


% 3 Number of Support Vectors


% 4 Confusion Matrix
%ToPrint = ['C: ', num2str(C), ' Train Confusion: ', trainConf, ' Validation Confusion: ', valConf];
%disp(ToPrint)

function [W, B, Obj] = train(X, Y, C)
    %Y = trLb;
    % QP Program
    % Start of Input parameters
    K = X'*X;
    %C = Cfact * ones(1, N);
    [D, N] = size(X);
    %K = ones(N, N);
    % End of Input parameters

    H = diag(Y)*K*diag(Y);
    %H = double(H);
    f = -1 * ones(1, N);
    l = zeros(N, 1);
    u = C * ones(N, 1);
    A = zeros(1, N);
    b = zeros(1, 1);
    Aeq = Y';
    Beq = zeros(1, 1);

    % QP Parameters
    [alpha, Obj] = quadprog(H, f, A, b, Aeq, Beq, l, u)
    
    max(alpha)
    min(alpha)
    % Q 3.2 - 3.3
    % find W: 
    size(alpha')
    size(Y)
    size(X)
    W = alpha * Y' * X';
    W = sum(W);
    %W = W';
    %W = Y' * diag(alpha) * X';
    W = (diag(alpha) * Y)' * X';
    %W = X * diag(Y) * alpha;
    W = W';
    
    YN = Y';
    B = YN - W' * X;
    B = min(B)/2;
    %B =  (max(B) - min(B)) / 2 + min(B);
    
end


function [Accuracy, Conf, SupportVectors, Obj] = predictFun(X, Y, W, B, C)
    [t, N] = size(X);
    K = length(unique(Y));
    % Values for all classes
    YPredict = sign(W'*X + B(1,1));
    YPredict = YPredict';
    % 1. Accuracy 
    Accuracy = sum(YPredict == Y) / N;
    WNorm = norm(W(:))^2;
    
    % Objective Value of SVM    
    Obj = Y'*(W'*X + B(1,1))';
    Obj = 1 - Obj';
    Obj = C * sum(Obj > 0) + WNorm / 2
    % Number of SVM
    
    YTemp = W'*X + B(1,1);
    [T, SupportVectors] = size(YTemp(YTemp >= -1 & YTemp <= 1))
    
    
    % Confusion Matrix
    Conf = confusionmat(Y, YPredict)
end


