load Processed_new.mat;
tmp_feature(:,8) = tmp_feature(:,8) /max(tmp_feature(:,8) );
[mappedX, mapping] = pca(tmp_feature, 0.62);

new = [mappedX tmp_outcome];