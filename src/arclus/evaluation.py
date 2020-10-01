import itertools
import math

import pandas as pd
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
