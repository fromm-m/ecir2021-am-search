"""Evaluation utilities."""
import math
from typing import Sequence

import pandas
import torch
from sklearn.metrics import f1_score


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


def mdcg_score(
    y_pred: Sequence[str],
    data: pandas.DataFrame,
    k: int,
) -> float:
    """
    Calculate the modified DCG score.

    :param y_pred:
        The predicted ranking, a sequence of premise IDs.
    :param data:
        The data, comprises at least columns "premise_id", "relevance", and "premiseClusterID_groundTruth".
    :param k: >0
        The cut-off parameter.

    :return:
        A non-negative scalar value.
    """
    df = pandas.DataFrame(data=dict(premise_id=y_pred))
    df.index.name = "position"
    df = df.reset_index()
    df = pandas.merge(data, df, how="inner", on="premise_id").sort_values(by='position')
    seen_clusters = set()
    gain = 0
    for i, relevance, cluster_id in zip(range(k), df["relevance"], df["premiseClusterID_groundTruth"]):
        if isinstance(cluster_id, str):
            if cluster_id in seen_clusters:
                relevance = 0
            seen_clusters.add(cluster_id)
        else:
            # only premises with relevance are assigned to clusters
            assert math.isnan(cluster_id)
            relevance = 0
        gain += relevance / math.log2(i + 2)
    return gain


def optimal_mdcg_score(
    data: pandas.DataFrame,
    k: int,
) -> float:
    """
    Calculate the optimal mDCG score for the given relevance/cluster data.

    :param data:
        The data, comprises at least columns "relevance", and "premiseClusterID_groundTruth".
    :param k:
        The cut-off parameter k.

    :return:
        The optimal value of mDCG@k.
    """
    # filter irrelevant
    data = data[(data["relevance"] > 0) & ~(data["premiseClusterID_groundTruth"].isna())]
    # get most relevant from each cluster
    data = data.groupby(by='premiseClusterID_groundTruth').agg(dict(relevance='max'))
    # sort by relevance
    data = data.sort_values(by='relevance', ascending=False).astype(float)
    gain = 0.
    for i, r in zip(range(k), data.values.flat):
        gain += r / math.log2(i + 2)
    return gain


def mndcg_score(
    y_pred: Sequence[str],
    data: pandas.DataFrame,
    k: int,
) -> float:
    """
    Calculate the modified NDCG score.

    :param y_pred:
        The predicted ranking, a sequence of premise IDs.
    :param data:
        The data, comprises at least columns "premise_id", "relevance", and "premiseClusterID_groundTruth".
    :param k: >0
        The cut-off parameter.

    :return:
        The modified NDCG score, a scalar between 0 and 1 (both incl.).
    """
    assert len(y_pred) == len(set(y_pred))
    opt_score = optimal_mdcg_score(data=data, k=k)
    if opt_score <= 0:
        return 0.
    score = mdcg_score(y_pred=y_pred, data=data, k=k)
    assert score <= opt_score
    return score / opt_score
