%W = load('./5/C0.1/C0.1_2.W.csv');
%W = load('./5/C10/C10_2.W.csv');

X = trD;
Y = trLb;
trainError = predictFun(X, Y, W)

X = valD;
Y = valLb;
valError = predictFun(X, Y, W)

function Accuracy = predictFun(X, Y, W)
    [t, N] = size(X);
    K = length(unique(Y));
    % Values for all classes
    YPredictAll = W'*X;

    % Get Class Index
    [YPVal, YPIndex] = max(YPredictAll);

    % map to print label
    YPredictionSet = unique(Y);
    IndexSet = 1:K;
    IndexToClassMap = containers.Map(IndexSet, YPredictionSet);

    % Get corresponding Index
    YPredict = zeros(N, 1);
    for i = 1:N
        YPredict(i) = IndexToClassMap(YPIndex(i));
    end

    Accuracy = sum(YPredict == Y) / N;

    WNorm = norm(W(:))^2
end
