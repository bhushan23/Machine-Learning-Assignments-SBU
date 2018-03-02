X = trD;
Y = trLb;
K = 2;
[D, N] = size(X);
W = zeros(D, K);

% Y label t W index mapping
YLabels = unique(Y);
Index = 1 : K;
ClassMap = containers.Map(YLabels, Index);

C = 0.1;
eta0 = 1;
eta1 = 100;
num_epochs = 2000;

iX = randperm(N);
%sX = X(iX);
%sY = Y(iX);
LossArray = zeros(num_epochs);
for epoch = 1 : num_epochs
    eta = eta0 / (eta1 + epoch);
    iX = randperm(N);
    curEpochLoss = 0;
    for i = iX %1 : N
        Xi = X(:, i);
        YMul = transpose(W) * Xi;
        YiIndex = ClassMap(Y(i));
        % Remove True label value
        YMul(YiIndex) = -Inf;
        
        % Handle case 
        [YHatVal, YHatIndex] = max(YMul);
        
        % Gradient cases
        % 1. True label
        MaxCondition = transpose(W(:,YHatIndex)) * Xi - transpose(W(:,YiIndex)) * Xi + 1;
        %curEpochLoss =  curEpochLoss + max(MaxCondition, 0);
        %LossArray = [LossArray, max(MaxCondition, 0)];
        WMul = W(:,YiIndex) / N;
        SGTrueLabel = WMul - C * Xi;
        if MaxCondition < 0
           SGTrueLabel = WMul; 
        end
        % 2. Y Hat
        WMul = W(:,YHatIndex) / N;
        SGHat = max(WMul - C * Xi, WMul);
        if MaxCondition < 0
            SGHat = WMul;
        end
        
        % 3. For rest of Wj's
        %tempW = (sum(W,2) - W(:,YHatIndex) - W(:,YiIndex)) / N;
        gradW = W / N;
        
        % Gradient matrix
        %gradW = repmat(tempW, 1, K);
        gradW(:,YiIndex) = SGTrueLabel;
        gradW(:,YHatIndex) = SGHat;
        
        % Update W 
        W = W - eta * gradW;
        
        % Compute loss
        normSum = 0;
        for tempK = 1:K
            normSum = normSum + norm(W(:,tempK));
        end
        
        curEpochLoss = curEpochLoss + normSum / (2*N) + C * max(0, MaxCondition);
    end
    % Append Loss
    LossArray(epoch) = curEpochLoss;
end

%plot(LossArray)
