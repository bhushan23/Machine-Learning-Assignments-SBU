
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



