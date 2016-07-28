
%% load data
data = load('pendigits-train.csv');
dataMatrix = data(: , 1:16);
row = size(data,1);
label = data(:,17);
length = size(dataMatrix,1);

% 4 fold validation
fold = 4;
Indices = crossvalind('Kfold', row, fold);
knn_ConfusionMat = zeros(10);
svm_ConfusionMat = zeros(10);
%% normalize
maxValue = max(dataMatrix);
minValue = min(dataMatrix);
range = maxValue-minValue;
normalizedMat = (dataMatrix-repmat(minValue,[length,1]))./(repmat(range,[length,1]));


%% knn cross validation
% k = 1;
%  k = 5;
k = 50;
knn_error = 0;
% 
% for f = 1:fold
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
% end
%    
%  knn_result = my_knn(TestSet,TrainingSet,label,k)

 
%% svm cross validation
 kernel = 'polynomial';

 kernel = 'rbf'; 

svm_error = 0;

for f = 1:fold
    test = (Indices==f);
    train = ~test;
    TrainingSet = normalizedMat(train,:);
    TrainingLabel = label(train);
    TestSet = normalizedMat(test,:);
    TestLabel = label(test);
    [~,svm_result] = multisvm(TrainingSet,TrainingLabel,TestSet,kernel);
    svm_result = svm_result-1;
    for i = 1 : size(svm_result,1)
        svm_ConfusionMat(svm_result(i)+1,TestLabel(i)+1) = svm_ConfusionMat(svm_result(i)+1,TestLabel(i)+1)+1;
    end
    svm_error = svm_error+ sum(svm_result ~= TestLabel);
end

%% draw an example instance 
% TrainingSet = normalizedMat;
% TestSet = normalizedMat(1,:);
% [models,~] = multisvm(TrainingSet,label,TestSet,kernel);
% m = 5;
% for i = 1:10
%     for j = 1:8
%         x(j) = models(i).SupportVectors(m,2*j-1);
%         y(j) = models(i).SupportVectors(m,2*j);   
%     end
%     subplot(2,5,i);
%     plot(x,y,'*-');
%     str=['Label',num2str(i-1)];
%     title(str);
% end

%% simple experiment 
% TrainingSet = normalizedMat;
% TrainingLabel = label;
% TrainingSet(2700:2750,:) =[];
% TrainingLabel(2700:2750) =[];
% for i = 2700:2750 %label = 7
%     knn_result_test(i-2699) = my_knn(normalizedMat(i,:),TrainingSet,TrainingLabel,k);
% end
% Cls2one = sum(knn_result_test ==1)%misclassified to 1
% 
% [~,svm_result_test] = multisvm(TrainingSet,TrainingLabel,normalizedMat(2700:2750,:),kernel);
% svm_result_test = svm_result_test-1;
% Cls2one2 = sum(svm_result_test ==1)



%% Predict
% nolabeldata = load('pendigits-test-nolabels.csv');
% nolabeldata = nolabeldata/100;
% TrainingSet = normalizedMat;
% TrainingLabel = label;
% [~,svm_result] = multisvm(TrainingSet,TrainingLabel,nolabeldata,kernel);
% svm_result = svm_result-1;


%   fid = fopen('output.txt', 'wt');
%    fprintf(fid, '%d\n', svm_result);
%    fclose(fid);
%% Transfer SVM

% source0 = load('0vs8Source.csv');
% source = source0(:,1:16);
% sourcelabel = source0(:,17);
% sourcelabel(find(sourcelabel==0))=-1;
% sourcelabel(find(sourcelabel==8))=1;
% 
% target0 = load('0vs8Target.csv');
% target = target0(:,1:16);
% targetlabel = target0(:,17);
% targetlabel(find(targetlabel==0))=-1;
% targetlabel(find(targetlabel==8))=1;
% 
% TestSet = load('0vs8TestNoLabels.csv');
% 
%  kernel = 'linear';
% % TestSet = target(10:30,:);
% % TestLabel = targetlabel(10:30);
% TrainingSet1 = target; 
% % TrainingSet1(10:30,:) = [];
% TrainingLabel1 = targetlabel; 
% % TrainingLabel1(10:30,:) = [];
% tmp = randperm(size(source,1));
% tmp = tmp(:,1:20)';
% TrainingSet2 = source(tmp,:);
% TrainingLabel2 = sourcelabel(tmp);
% TrainingSet = [TrainingSet1;TrainingSet2];
% TrainingLabel = [TrainingLabel1;TrainingLabel2];
% % TrainingSet = source;
% % TrainingLabel = sourcelabel;
% [~,svm_result] = multisvm(TrainingSet,TrainingLabel,TestSet,kernel); %learn from source.
% 
% output = svm_result;
% output(output==1) = 0;
% output(output==2) = 8;
%   fid = fopen('output2.txt', 'wt');
%    fprintf(fid, '%d\n', output);
%    fclose(fid);

