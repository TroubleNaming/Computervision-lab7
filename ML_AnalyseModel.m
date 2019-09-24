function [confmat, acc, prec, rec, f1score] = ML_AnalyseModel(pred_,gt_)
    confmat = zeros(10,10);
    TPTN = 0;
    for i = 1:10
        for j = 1:10
            msk_1 = pred_==(i-1);
            msk_2 = gt_==(j-1);
            msk_t = msk_1 & msk_2;
            confmat(i,j) = sum(msk_t);
            if j==i
               TPTN = TPTN+ confmat(i,j);
            end
        end
    end
    acc = TPTN/length(gt_);
    for class = 1:10
        TP = confmat(class,class);
        FP = sum(confmat(class,:))-TP;
        FN = sum(confmat(:,class))-TP;
        TN = sum(confmat,'all')-TP-FP-FN;
        prec(class) = TP/(TP+FP);
        rec(class) = TP/(TP+FN);
        f1score(class) = 2*rec(class)*prec(class)/(rec(class)+prec(class));
    end
end
