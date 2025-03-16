import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score




def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label==1) & (predicted_label==1)))
    tn = float(np.sum((true_label==0) & (predicted_label==0)))
    p = float(np.sum(true_label==1))
    n = float(np.sum(true_label==0))

    return (tp * (n/p) +tn) / (2*n)


def eval_private(results, truths):
    test_preds = results.cpu().detach().numpy()
    test_truth = truths.cpu().detach().numpy()
    # print(f"test_preds:{test_preds.shape}")
    # print(f"test_truth:{test_truth.shape}")
    test_preds_i = np.argmax(test_preds,axis=1)
    test_truth_i = test_truth
    f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
    acc = accuracy_score(test_truth_i, test_preds_i)
    print(f"  - f1 SCORE: {f1}")
    print(f"  - ACCURACY: {acc}")
    return acc, f1

