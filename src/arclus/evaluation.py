"""Evaluation methods."""
import itertools
import math

import numpy
import pandas as pd
import torch
from sklearn.metrics import f1_score


def mdcg(
    query_claim_ids: numpy.ndarray,
    predicted_rankings: numpy.ndarray,
    premise_cluster_ids: numpy.ndarray,
    premise_relevance: numpy.ndarray,
) -> numpy.ndarray:
    """
    Compute modified DCG.

    :param query_claim_ids: shape: (num_queries,), dtype: int
        The query claim ids.
    :param predicted_rankings: shape: (num_queries, k), dtype: int
        The predicted rankings, comprising ordered sequences of premise IDs.
    :param premise_cluster_ids: shape: (num_premises,), dtype: int
        The cluster ID for each premise.
    :param premise_relevance: shape: (num_claims, num_premises), dtype: float (or int)
        The relevance of each premise for a claim.

    :return: shape: (num_queries,)
        The modified NCG for each query.
    """
    # allocate result array
    score = numpy.zeros_like(query_claim_ids, dtype=numpy.float64)

    # iterate
    for i in range(predicted_rankings.shape[1]):
        # get relevance
        relevance = premise_relevance[query_claim_ids, predicted_rankings[:, i]]

        # discard relevance if not first occurrence of cluster
        if i > 0:
            current_cluster = premise_cluster_ids[predicted_rankings[:, i]]
            previous_clusters = premise_cluster_ids[predicted_rankings[:, :i]]
            novel = (current_cluster[:, None] == previous_clusters).any(axis=-1)
            relevance = relevance * novel.astype(relevance.dtype)

        # discount
        score += relevance / numpy.log2(i + 2)

    return score


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
        if i + 1 > 2:
            gain = gain + (value / math.log2(i + 1))
        else:
            gain = gain + value
    return gain


def ndcg_score(y_score: [int], y_true: [int], k: int) -> int:
    """
    Calculates normalized discounted cumulative gain (nDCG) for a prediction and groundtruth ranking.
    :param y_score: shape: (number_clusters)
        The rankings of all prediction clusters
    :param y_true: shape: (number_clusters)
        The rankings of all groundtruth clusters
    :return: gain:
        The normalized discounted cumulative gain (DCG)
    """
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


def task_b(ordered_clusters: [pd.DataFrame]) -> [int]:
    """
    Calculates the ranking given the predicted_clusters, as the representative the longest premise is chosen.
    :param ordered_clusters: shape: (number_predicted_clusters)
        The ordered_clusters containing the premises.
    :return: ranking:
        The ranking for all predicted clusters ordered by the cluster_id.
    """
    ranking = []
    seen_gt_clusters = set()
    for cluster in ordered_clusters:
        # calculate max_length of all premises in the cluster i
        max_length = cluster.resultClaimsPremiseText.str.len().max()
        # search for the longest premise in cluster i
        premise_represent = cluster.loc[cluster["resultClaimsPremiseText"].str.len() == max_length]
        if premise_represent.empty or math.isnan(max_length):
            ranking.append(0)
        elif str(premise_represent["premiseClusterID_groundTruth"].values[0]) in seen_gt_clusters:
            ranking.append(0)
        else:
            ranking.append(premise_represent["relevance"].values[0])
            seen_gt_clusters.add(str(premise_represent["premiseClusterID_groundTruth"].values[0]))
    return ranking


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
            if premise_represent["premiseClusterID_groundTruth"].values[0] in seen_gt_clusters:
                ranking.append(0)
            else:
                ranking.append(premise_represent["relevance"].values[0])
                seen_gt_clusters.add(premise_represent["premiseClusterID_groundTruth"].values[0])
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
        ranking.append(premise_represent_gt["relevance"].values[0])
    return ranking
