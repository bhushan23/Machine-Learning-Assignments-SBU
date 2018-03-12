Xtest = tstD;
YtestPred = W' * Xtest;
YPredictionSet = unique(Y);

predictTest(Xtest, W, YPredictionSet)

function predictTest(X, W, YPredictionSet)
    [t, N] = size(X);
    K = 10; % FOR 3.2
    % Values for all classes
    YPredictAll = W'*X;

    % Get Class Index
    [YPVal, YPIndex] = max(YPredictAll);

    % map to print label
    %YPredictionSet = unique(Y);
    IndexSet = 1:K;
    IndexToClassMap = containers.Map(IndexSet, YPredictionSet);

    % Get corresponding Index
    YPredict = zeros(N, 1);
    for i = 1:N
        YPredict(i) = IndexToClassMap(YPIndex(i));
    end

    T = readtable('./test/SampleSubmission.csv');
    Z = table(YPredict);
    T(:,{'Class'}) = Z;
    writetable(T, './test/Submission5.csv');
    %csvwrite('test/submission4.csv', YPredict)
    %Accuracy = sum(YPredict == Y) / N;

    %WNorm = norm(W(:))^2
end
