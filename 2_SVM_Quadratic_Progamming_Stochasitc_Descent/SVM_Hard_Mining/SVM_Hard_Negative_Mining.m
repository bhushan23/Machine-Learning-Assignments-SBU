[trD, trLb, valD, valLb, trRegs, valRegs] = HW2_Utils.getPosAndRandomNeg();

C = 10;
num_epochs = 10;    

[W, B, OFunc, APFunc] = hardMining(trD, trLb, valD, valLb, num_epochs, C, 1, ubAnno);

%[W, B] = hardMining(valD, valLb, num_epochs, C, 0)

predictFun(valD, valLb, W, B)
predictFun(trD, trLb, W, B)
%plot(APFunc);
plot(OFunc);

function [W, B, OFunc, APFunc] = hardMining(trD, trLb, valD, valLb, num_epochs, C, IsTrain, ubAnno) 
% IsTrain == 1: Training Otherwise Val
    [D, N] = size(trD);
    posSize = size(trLb(trLb == 1));

    YPos = ones(posSize(1,1), 1);
    YNeg = -1 * ones(N-posSize(1,1), 1);

    posD = trD(:,1:posSize);
    negD = trD(:,posSize+1:N);
    
    size(posD);
    size(negD);
    size(YPos);
    size(YNeg);

    X = trD;
    Y = trLb;
    size(X);

    [W, B] = trainQ3(X, Y, C);

    A = W'*X + B;
    path = './trainIms/';
    if IsTrain ~= 1
        path = './valIms/';
    end
    
    APFunc = zeros(num_epochs, 1);
    OFunc = zeros(num_epochs, 1);
    for i=1:num_epochs
        negDLabel = W'*negD + B;
        
        Temp = negDLabel >= -1 & negDLabel <= 1;
        A = negD(:, Temp);
        negD = A;
        
        files = dir(fullfile(path, '*.jpg'));
        nums = length(files);
        reacts = cell(1, nums);
        newDataCount = 0;
        for k = 1:length(files)
            fPath = strcat(path,files(k).name);
            image = imread(fPath);
            reacts{k} = HW2_Utils.detect(image, W, B, 0);

            [imH, imW,~] = size(image);

            % How many negative scores are present
            negIndex = sum(reacts{k}(end, :) < 0);
            
            %if negIndex == 0
            %   disp('NEGIDEX 0 -- SKIPING')
            %   continue
            %end
               
            % Access last 10/last few
            size(reacts{k});
            
            currentRect = reacts{k}(:, end-negIndex+1: end);            
            posRect =  reacts{k}(:, 1:end-negIndex);

            % remove random rects that do not lie within image boundaries
            badIdxs = or(currentRect(3,:) > imW, currentRect(4,:) > imH);
            negEx = currentRect(:,~badIdxs);
                        
            % Remove random rects that overlap more than 30% with an annotated upper body
            if negIndex < size(reacts{k}, 2)
                ubs = ubAnno{k};
                for j=1:size(ubs,2)
                    overlap = HW2_Utils.rectOverlap(negEx, ubs(:,j));                    
                    negEx = negEx(:, overlap < 0.3);
                    if isempty(negEx)
                        break;
                    end
                end
            end

            size(negEx);
            [D_i, R_i] = deal(cell(1, size(negEx,2)));
            
            % Generate feature vectors
            for j=1:size(negEx, 2)
                imReg = image(negEx(2,j):negEx(4,j), negEx(1,j):negEx(3,j),:);
                imReg = imresize(imReg, HW2_Utils.normImSz);
                R_i{j} = imReg;
                D_i{j} = HW2_Utils.cmpFeat(rgb2gray(imReg));  
            end
            
            mat = cell2mat(D_i);
            %disp('MAT');
            size(mat);
            size(negD);
            size(D_i);
            
            negD = horzcat(negD, mat);
            newDataCount =  newDataCount + size(mat, 2);
            
            if newDataCount > 1000
                break;
            end
        end   
        
        X = horzcat(posD, negD);
        Y = vertcat(YPos, -1 * ones(size(negD, 2), 1));
        disp('Updated NEGD')
        size(negD,2)
        size(X)
        size(Y)
        [W, B, FObj] = trainQ3(X, Y, C);
        % Get VAL AP
        HW2_Utils.genRsltFile(W, B, 'val', 'result');
        [ap, prec, rec] = HW2_Utils.cmpAP('result', 'val');
        APFunc(i) = ap;
        OFunc(i) = -1 * FObj;
        ToPrint = ['Iteration: ', num2str(i),  ' C: ', num2str(C), ' Obj: ', num2str(-1 * FObj), ' AP: ', num2str(ap)];
        disp(ToPrint);
        predictFun(valD, valLb, W, B)
        predictFun(trD, trLb, W, B)
    end
end

function [NegIndex] = firstNegIndex(Input)
    bits = Input < 0;
    NegIndex = sum(bits);
end

function [W, B, Obj] = trainQ3(X, Y, C)
    %Y = trLb;
    % QP Program
    % Start of Input parameters
    Y = double(Y);
    X = double(X);
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
    [alpha, Obj] = quadprog(H, f, A, b, Aeq, Beq, l, u);
    
    max(alpha);
    min(alpha);
    % Q 3.2 - 3.3
    % find W: 
    size(alpha');
    size(Y);
    size(X);
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

function [Accuracy] = predictFun(X, Y, W, B)
    [t, N] = size(X);
    %K = length(unique(Y));
    % Values for all classes
    YPredict = sign(W'*X + B(1,1));
    YPredict = YPredict';
    % 1. Accuracy 
    Accuracy = sum(YPredict == Y) / N;
    WNorm = norm(W(:))^2;
end


function W = train4(X, Y, W, C, num_epochs, K)
    %K = length(unique(Y)) %2;
    [D, N] = size(X);
    % Y label t W index mapping
    YLabels = unique(Y);
    Index = 1 : K;

    ClassMap = containers.Map(YLabels, Index);
    % C = 10;
    eta0 = 1;
    eta1 = 100;
    %num_epochs = 2000;

    iX = randperm(N);
    %sX = X(iX);
    %sY = Y(iX);
    LossArray = zeros(num_epochs, 1);
    for epoch = 1 : num_epochs
        eta = eta0 / (eta1 + epoch);
        iX = randperm(N);
        curEpochLoss = 0.0;
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
            MaxCondition = transpose(W(:,YHatIndex)) * Xi - transpose(W(:,YiIndex)) * Xi + 1.0;
            %curEpochLoss =  curEpochLoss + max(MaxCondition, 0);
            %LossArray = [LossArray, max(MaxCondition, 0)];
            WMul = W(:,YiIndex) / N;
            SGTrueLabel = WMul;
            if MaxCondition > 0
               SGTrueLabel = WMul - C * Xi; 
            end
            % 2. Y Hat
            WMul = W(:,YHatIndex) / N;
            SGHat = WMul;
            if MaxCondition > 0
                SGHat = WMul + C*Xi;
            end

            % 3. For rest of Wj's
            %tempW = (sum(W,2) - W(:,YHatIndex) - W(:,YiIndex)) / N;
            gradW = W / N;

            % Gradient matrix
            gradW(:,YiIndex) = SGTrueLabel;
            gradW(:,YHatIndex) = SGHat;

            % Update W 
            W = W - eta * gradW;

            % Compute loss
            normSum = norm(W(:))^2;
            MaxCondition = transpose(W(:,YHatIndex)) * Xi - transpose(W(:,YiIndex)) * Xi + 1.0;
            curEpochLoss = curEpochLoss + normSum / (2*N) + C * max(0, MaxCondition);
        end
        % Append Loss
        LossArray(epoch) = curEpochLoss;
    end
    plot(LossArray)
end
