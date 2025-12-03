import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(y_true, y_pred, average='weighted'):
    """
    Calculate evaluation metrics for classification tasks.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        average (str): Averaging method for multi-class metrics ('weighted', 'macro', 'micro').

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1-score.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average),
        'recall': recall_score(y_true, y_pred, average=average),
        'f1_score': f1_score(y_true, y_pred, average=average)
    }
    return metrics