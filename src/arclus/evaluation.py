from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import torch


def accuracy(
    pred_y,
    labels,
    prob=True
) -> torch.tensor:
    if prob:
        _, pred_y = torch.max(pred_y, dim=1)
    return torch.tensor(torch.sum(pred_y == labels).item() / len(pred_y))


def f1_macro(
    pred_y,
    labels,
    prob=True
) -> float:
    if prob:
        _, pred_y = torch.max(pred_y, dim=1)

    f1_macro = f1_score(y_pred=pred_y, y_true=labels, average='macro')
    return f1_macro


def class_metrics(
    pred_y,
    labels,
    prob=True
):
    if prob:
        _, pred_y = torch.max(pred_y, dim=1)
    return classification_report(labels, pred_y)
