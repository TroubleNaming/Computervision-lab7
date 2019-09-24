clear
clc

load('MNIST_dataset.mat');
tst = data_test(:,:,1:50);
trn = data_train(:,:,1:500);
lbl_tst = labels_test(1:50);
lbl_trn = labels_train(1:500);

% fords to be 6
K = 6;
[ indices_train, indices_val ] = ML_CrossVal_KFold( K, length(lbl_trn) );
ACC = zeros(1,K);
for i = 1:K
    train_lbl = lbl_trn(indices_train(i,:));
    train_data = trn(:,:,indices_train(i,:));
    data_features = reshape(train_data,28*28,450);
    model = fitcknn(data_features',train_lbl);
    val_lbl = lbl_trn(indices_val(i,:));
    val_data = trn(:,:,indices_val(i,:));
    val_features = reshape(val_data,28*28,450);
    pred_ = predict(model, val_features');
    [confmat, acc, prec, rec, f1score] = ML_AnalyseModel(pred_,val_lbl);
end
