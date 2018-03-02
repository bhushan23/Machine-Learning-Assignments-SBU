path = "5/C0.1/C0.1_2.W.csv"
csvwrite(path, W);
plot(LossArray)
xlabel('Iteration')
ylabel('Loss')
title('Loss for 2000 iterations with C=0.1')
