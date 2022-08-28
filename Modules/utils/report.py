import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd


def sklearn_classification_report(y_true, y_pred):
    label2id = {"CN": 0, "MCI": 1, "AD": 2}
    print(classification_report(y_true, y_pred, labels=[0, 1, 2],  target_names=list(label2id.keys())))
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label2id.keys()))
    disp.plot();
    
    
def custom_classification_report(y_true, y_pred, show_cm_matrix=False):
    label2id = {"CN": 0, "MCI": 1, "AD": 2}
    metrics = {}
    average_metrics = {}

    cm = confusion_matrix(y_true, y_pred)
    
    if show_cm_matrix:
        cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=list(label2id.keys()))
        disp.plot();
    
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    accuracy = (TP + TN) / (TP + FP + FN + TN)
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    precision = TP / (TP + FP)
    recall = sensitivity
    f1_score = 2 * ((precision * recall) / (precision + recall))
    
    # metrics['Accuracy'] = accuracy.round(2)
    metrics['Sensitivity'] = sensitivity.round(2)
    metrics['Specificity'] = specificity.round(2)
    metrics['Precision'] = precision.round(2)
    # metrics['Recall'] = recall.round(2)
    metrics['F1-score'] = f1_score.round(2)
    
    df = pd.DataFrame.from_dict(metrics)
    df.index = list(label2id.keys())
    print(df, "\n")
    
    # average_metrics['Macro-average Accuracy'] = metrics['Accuracy'].mean().round(2)
    average_metrics['Macro-average Sensitivity'] = metrics['Sensitivity'].mean().round(2)
    average_metrics['Macro-average Specificity'] = metrics['Specificity'].mean().round(2)
    average_metrics['Macro-average Precision'] = metrics['Precision'].mean().round(2)
    # average_metrics['Macro-average Recall'] = metrics['Recall'].mean().round(2)
    average_metrics['Macro-average F1-score'] = metrics['F1-score'].mean().round(2)
    
    df = pd.DataFrame.from_dict(average_metrics, orient='index', columns=['Values'])
    print(df, "\n")