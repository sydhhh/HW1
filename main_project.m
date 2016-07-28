
%% load data
%data = load('pendigits-train.csv');
load Processed_new.mat;
% load new.mat
dataMatrix = tmp_feature;
% dataMatrix = new(:,1:10);
row = size(dataMatrix,1);
label = tmp_outcome-1;
% label = new(:,11)-1;
length = size(dataMatrix,1);

% 4 fold validation
fold = 4;
Indices = crossvalind('Kfold', row, fold);
knn_ConfusionMat = zeros(5);
svm_ConfusionMat = zeros(5);
%% normalize
maxValue = max(dataMatrix);
minValue = min(dataMatrix);
range = maxValue-minValue;
normalizedMat = (dataMatrix-repmat(minValue,[length,1]))./(repmat(range,[length,1]));


%% knn cross validation
% % k = 1;
% %  k = 5;
%  k = 50; 
% %  k = 500;
% 
% knn_error = 0;
% 
% for f = 1:fold
%     f
%     test = (Indices==f);
%     train = ~test;
%     TrainingSet = normalizedMat(train,:);
%     TrainingLabel = label(train);
%     TestSet = normalizedMat(test,:);
%     TestLabel = label(test);
%     knn_result = zeros(1,size(TestLabel,1));
%     for i = 1: size(TestLabel,1)
%         knn_result(i) = my_knn(TestSet(i,:),TrainingSet,TrainingLabel,k);
%         knn_ConfusionMat(knn_result(i)+1,TestLabel(i)+1) = knn_ConfusionMat(knn_result(i)+1,TestLabel(i)+1)+1;
%         if (knn_result(i)~=TestLabel(i))
%             knn_error = knn_error+1;
%         end
%     end
%     
% end
%  accuracy = 1-knn_error/row;
% %  knn_result = my_knn(TestSet,TrainingSet,label,k)

 
%% svm cross validation
    normalizedMat = tmp_feature;
    label = tmp_outcome;

    TrainingSet = normalizedMat;
    TrainingLabel = label;
    Traing_acc = 0;
    Test_acc = 0;
    for f = 1:fold
        test = (Indices==f);
        train = ~test;
        TrainingSet = normalizedMat(train,:);
        TrainingLabel = label(train);
        TestSet = normalizedMat(test,:);
        TestLabel = label(test);
        cmd = [' -t 0 '];
        model = svmtrain(TrainingLabel,TrainingSet,cmd);
        [~,acctrain,~] = svmpredict(TrainingLabel,TrainingSet,model)
        [~,acctest,~] = svmpredict(TestLabel,TestSet,model)
        Traing_acc = Traing_acc+acctrain(1)/4;
        Test_acc = Test_acc + acctest(1)/4;
    end
    



