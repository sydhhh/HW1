function relustLabel = my_knn(input,data,label,k)
    [row , col] = size(data);
    diff = repmat(input,[row,1]) - data ;
    distance = sqrt(sum(diff.^2,2));
    [ sorted, index] = sort(distance,'ascend');
    relustLabel = mode(label(index(1:k)));
end
