"""Evaluation utilities."""
import math
from typing import Collection, Sequence, Union

import pandas
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from arclus.models.base import RankingMethod
from arclus.utils import load_assignments_with_numeric_relevance


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
    a = data[data["relevance"] > 0].groupby(by='premiseClusterID_groundTruth').agg(dict(relevance='max')).sort_values(by='relevance', ascending=False).astype(float)
    gain = 0.
    for i, r in zip(range(k), a.values.flat):
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
    opt_score = optimal_mdcg_score(data=data, k=k)
    if opt_score <= 0:
        return 0.
    score = mdcg_score(y_pred=y_pred, data=data, k=k)
    assert score <= opt_score
    return score / opt_score


def evaluate_ranking_method(
    method: RankingMethod,
    k: Union[int, Collection[int]],
) -> pandas.DataFrame:
    """
    Evaluate a ranking method with mnDCG@k.

    :param method:
        The ranking method.
    :param k: > 0
        (Potentially multiple) cut-off parameter(s).

    :return:
        A dataframe with columns ["claim_id", "k", "mnDCG"].
    """
    # Input normalization
    if isinstance(k, int):
        k = [k]
    # load assignments
    df = load_assignments_with_numeric_relevance()
    # keep only relevant columns
    df = df.loc[:, ["claim_id", "premise_id", "relevance", "premiseClusterID_groundTruth"]]
    # iterate over all claims
    result_data = []
    for claim_id, queries in tqdm(df.groupby(by="claim_id"), unit='claim'):
        for kk in k:
            # predict ranking
            predicted_ranking = method.rank(
                claim_id=claim_id,
                premise_ids=queries["premise_id"].to_numpy(dtype=str),
                k=kk,
            )
            # evaluate ranking
            score = mndcg_score(y_pred=predicted_ranking, data=queries, k=kk)
            result_data.append((claim_id, kk, score))
    return pandas.DataFrame(data=result_data, columns=["claim_id", "k", "mnDCG"])


def evaluate_ranking_method_related_work(
    method: str,
    k: Union[int, Collection[int]],
    column: str,
) -> pandas.DataFrame:
    """
    Evaluate a ranking method with mnDCG@k, whose results are already contained in the dataframe.

    :param method:
        The ranking method. Must be in {"first512Tokens", "slidingWindow", "sentences"}.
    :param k: > 0
        (Potentially multiple) cut-off parameter(s).
    :param column:
        The column prefix for the score column. The final column is composed as "{column}_{method}".

    :return:
        A dataframe with columns ["claim_id", "k", "mnDCG"].
    """
    # Input normalization
    if isinstance(k, int):
        k = [k]
    # load assignments
    df = load_assignments_with_numeric_relevance()
    assert method in {'sentences', 'slidingWindow', 'first512Tokens'}
    # keep only relevant columns
    method_col = f"{column}_{method}"
    cluster_col = f"premiseClusterID_{method}"
    df = df.loc[:, ["claim_id", "premise_id", "relevance", "premiseClusterID_groundTruth", method_col, cluster_col]]
    # iterate over all claims
    result_data = []
    for claim_id, queries in tqdm(df.groupby(by="claim_id"), unit='claim'):
        for kk in k:
            predicted_ranking = []
            seen_cluster = set()
            for _, row in queries.sort_values(by=method_col, ascending=False).iterrows():
                cluster_id = row[cluster_col]
                if cluster_id not in seen_cluster:
                    predicted_ranking.append(row['premise_id'])
                    seen_cluster.add(cluster_id)
            predicted_ranking = predicted_ranking[:kk]
            # evaluate ranking
            score = mndcg_score(y_pred=predicted_ranking, data=queries, k=kk)
            result_data.append((claim_id, kk, score))
    return pandas.DataFrame(data=result_data, columns=["claim_id", "k", "mnDCG"])
