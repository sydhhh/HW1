function [models,result] = multisvm(TrainingSet,label,input,kernel)

u=unique(label);
numClasses=length(u);
result = zeros(length(input(:,1)),1);


for k=1:numClasses
    OnevAll=(label==u(k));
    models(k) = svmtrain(TrainingSet,OnevAll,'kernel_function',kernel,'autoscale',false); %%different models
end
%classify 
for j=1:size(input,1)
    for k=1:numClasses
        if(svmclassify(models(k),input(j,:))) 
            break;
        end
    end
    result(j) = k;
end