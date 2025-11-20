import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def compute_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = (torch.sigmoid(y_pred) > 0.5).cpu().numpy()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
