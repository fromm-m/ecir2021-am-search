from sklearn.metrics import f1_score
import torch


def accuracy(
    pred_y,
    labels,
) -> torch.Tensor:
    return (pred_y == labels).sum().item() / len(pred_y)


def f1_macro(
    pred_y,
    labels,
) -> float:
    return f1_score(y_pred=pred_y, y_true=labels, average='macro')
