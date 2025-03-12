import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
import pandas as pd


def calc_other(y_true, y_score, kind):
    cm = confusion_matrix(y_true, y_score)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    if kind == 'sensitivity':
        return sensitivity
    elif kind == 'specificity':
        return specificity
    elif kind == 'precision':
        return precision
    else:
        return f1


def bootstrap_data(y, pred, classes, bootstraps=2000, fold_size=200, data_type='auc', p=0.5):
    statistics = np.zeros((len(classes), bootstraps))
    for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred'])
        # df.
        df.loc[:, 'y'] = y
        df.loc[:, 'pred'] = pred
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            pos_sample = df_pos.sample(n=int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n=int(fold_size * (1 - prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            if data_type == 'auc':
                score = roc_auc_score(y_sample, pred_sample)
            elif data_type == 'acc':
                predict_list = np.zeros_like(pred_sample)
                predict_list[pred_sample >= p] = 1
                score = sum(y_sample == predict_list) / len(y_sample)
            else:
                predict_list = np.zeros_like(pred_sample, dtype=int)
                predict_list[pred_sample >= p] = 1
                score = calc_other(y_sample, predict_list, data_type)
            statistics[c][i] = score
    up = np.percentile(statistics, 95, axis=1)[1]
    low = np.percentile(statistics, 5, axis=1)[1]
    # print(np.percentile(statistics, 95, axis=1), np.max(statistics, axis=1))
    return low, up


class Metrics:
    def __init__(self, labels, predictions, probs):
        self.labels = labels
        self.predictions = predictions
        self.probs = probs
        cm = confusion_matrix(self.labels, self.predictions)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
        self.results = [sensitivity, specificity, precision, f1, cm]

    def accuracy(self):
        return accuracy_score(self.labels, self.predictions)

    def confusion_matrix(self):
        return self.results[4]

    def AUC(self):
        fpr, tpr, _ = roc_curve(self.labels, self.probs)
        return auc(fpr, tpr)

    def sensitivity(self):
        return self.results[0]

    def specificity(self):
        return self.results[1]

    def precision(self):
        return self.results[2]

    def f1(self):
        return self.results[3]

    def roc_curve(self):
        fpr, tpr, auc_value = roc_curve(self.labels, self.probs)
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_value:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
