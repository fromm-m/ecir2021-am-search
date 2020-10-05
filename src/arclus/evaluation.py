import itertools
import math
from typing import Collection, Sequence, Union

import pandas
import pandas as pd
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from arclus.models.baselines import RankingMethod
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


def evaluate_premises(premises):
    """
    Calculates the ranking given the predicted_clusters, as the representative the longest premise is chosen.
    :param ordered_clusters: shape: (number_predicted_clusters)
        The ordered_clusters containing the premises.
    :return: ranking:
        The ranking for all predicted clusters ordered by the cluster_id.
    """
    ranking = []
    seen_gt_clusters = set()
    for index, row in premises.iterrows():
        if str(row["premiseClusterID_groundTruth"]) in seen_gt_clusters:
            ranking.append(0)
        else:
            ranking.append(row["relevance"])
            seen_gt_clusters.add(str(row["premiseClusterID_groundTruth"]))
    return ranking


def dcg(arr: [int]) -> int:
    """
    Calculates discounted cumulative gain (DCG) for a given ranking.
    :param arr: shape: (number_clusters)
        The rankings of all clusters
    :return: gain:
        The discounted cumulative gain
    """
    gain = 0.
    for i, value in enumerate(arr):
        gain = gain + (value / math.log2(i + 2))
    return gain


def ndcg_score(y_score: [int], y_true: [int], k: int, pad: bool) -> int:
    """
    Calculates normalized discounted cumulative gain (nDCG) for a prediction and groundtruth ranking.
    :param y_score: shape: (number_clusters)
        The rankings of all prediction clusters
    :param y_true: shape: (number_clusters)
        The rankings of all groundtruth clusters
    :param k:
        Decides how long the maximal ranking is.
    :param pad:
        If true the rankings are padded with 0s until they reach length k.
    :return: gain:
        The normalized discounted cumulative gain (DCG)
    """
    if pad:
        while len(y_score) < k:
            y_score.append(0)
        while len(y_true) < k:
            y_true.append(0)
    return dcg(y_score[:k]) / dcg(y_true[:k])


def split_clusters(df: pd.DataFrame, cluster_ids: [int], condition: str) -> [pd.DataFrame]:
    """
    Splits all premises for a given claim according to their assigned clusters.
    :param df: shape: (number_premises)
        The premises of a given claim
    :param cluster_ids: shape: (number_clusters)
        The cluster_ids of all clusters
    :param condition:
        The condition (column name) on which the clusters should be split.
    :return: gain:
        The normalized discounted cumulative gain (DCG)
    """
    clusters_list = []
    # iterate over all predicted clusters starting with lowest id
    for cluster_id in cluster_ids:
        # only select the premises which are in cluster i
        cluster = df.loc[df[condition] == cluster_id]
        clusters_list.append(cluster)
    return clusters_list


def find_cluster_representatives(ordered_clusters: [pd.DataFrame]) -> [str]:
    """
    Given the predicted_clusters, return as list of chosen cluster representatives (premises)
    :param ordered_clusters: shape: (number_predicted_clusters)
        The ordered_clusters containing the premises.
    :return: ranking:
        The ranking for all predicted clusters ordered by the cluster_id.
    """
    premises = pd.DataFrame([])
    for cluster in ordered_clusters:
        # search for the longest premise in cluster i
        premise_texts = cluster.premise_text.to_numpy(dtype=str)
        max_length = len(max(premise_texts, key=len))
        premise_represent = cluster.loc[cluster.premise_text.str.len() == max_length]
        premises = premises.append(premise_represent)
    return premises


def task_a(df_temp: pd.DataFrame, ordered_clusters: [pd.DataFrame]) -> [[int]]:
    """
        Calculates the ranking given the predicted_clusters, all combinations of premises are tried.
        :param ordered_clusters: shape: (number_predicted_clusters)
            The ordered_clusters containing the premises.
        :return: ranking:
            The ranking for all predicted clusters ordered by the cluster_id.
        """
    list = []
    for cluster in ordered_clusters:
        premise_ids = []
        for index, row in cluster.iterrows():
            premise_ids.append(row["resultClaimsPremiseID"])
        list.append(premise_ids)

    all_rankings = []
    for combination in itertools.product(*list):
        ranking = []
        seen_gt_clusters = set()
        for premise_id in combination:
            premise_represent = df_temp.loc[df_temp["resultClaimsPremiseID"] == premise_id]
            if premise_represent["premiseClusterID_groundTruth"].to_numpy(dtype=str)[0] in seen_gt_clusters:
                ranking.append(0)
            else:
                ranking.append(premise_represent.relevance.to_numpy(dtype=int)[0])
                seen_gt_clusters.add(premise_represent["premiseClusterID_groundTruth"].to_numpy(dtype=int)[0])
        all_rankings.append(ranking)
    return all_rankings


def best_ranking(ordered_clusters: [pd.DataFrame]) -> [int]:
    """
    Calculates the best ranking given the groundtruth_clusters, as the representative the premise with the
    highest relevance is chosen.
    :param ordered_clusters: shape: (number_gt_clusters)
        The ordered_clusters containing the premises.
    :return: ranking:
        The ranking for all groundtruth clusters ordered by the cluster_id.
    """
    ranking = []
    for cluster in ordered_clusters:
        # calculate max_relevance of all premises in the cluster i
        max_relevance = cluster.relevance.max()
        # search for the maximal relevant premise in cluster i
        premise_represent_gt = cluster.loc[cluster["relevance"] == max_relevance]
        # return the relevance value of the most relevant premise in c
        ranking.append(premise_represent_gt.relevance.to_numpy(dtype=int)[0])
    return ranking


def mdcg_score(
    y_pred: Sequence[str],
    data: pandas.DataFrame,
    k: int,
) -> float:
    df = pandas.DataFrame(data=dict(premise_id=y_pred))
    df.index.name = "position"
    df = df.reset_index()
    df = pandas.merge(data, df, how="inner", on="premise_id").sort_values(by='position')
    seen_clusters = set()
    gain = 0
    for i, (row_id, row) in zip(range(k), df.iterrows()):
        relevance = row.relevance
        cluster_id = row.premiseClusterID_groundTruth
        if isinstance(cluster_id, str):
            if cluster_id in seen_clusters:
                relevance = 0
            seen_clusters.add(cluster_id)
        else:
            assert math.isnan(cluster_id)
        gain += relevance / math.log2(i + 2)
    return gain


def optimal_mdcg_score(
    data: pandas.DataFrame,
    k: int,
) -> float:
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
