import pandas as pd
import numpy as np
import math
import argparse
from arclus.settings import PREP_ASSIGNMENTS_TEST


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


def main():
    """
    Calculates normalized discounted cumulative gain (nDCG) for all queries.
    """
    parser = argparse.ArgumentParser(description='Pre-compute BERT features.')
    parser.add_argument('--algorithm', default='premiseClusterID_first512Tokens',
                        choices=['premiseClusterID_first512Tokens', 'premiseClusterID_slidingWindow',
                                 'premiseClusterID_sentences'], help='The algorithm which should be evaluated.')
    args = parser.parse_args()
    algorithm = args.algorithm
    df = pd.read_csv(PREP_ASSIGNMENTS_TEST, sep=";")

    # set the relevance to the according value (cf. paper)
    df['relevance'].loc[(df['relevance'] == "notRelevant")] = 0
    df['relevance'].loc[(df['relevance'] == "yesRelevant")] = 1
    df['relevance'].loc[(df['relevance'] == "yesVeryRelevant")] = 2

    ndcg_list = []
    # Iterate over each claim
    for query in df["queryClaimID"].unique():
        # select all premises for a given claim
        df_temp = df.loc[df["queryClaimID"] == query]

        # unique predicted cluster_ids for a given claim
        ordered_predicted_cluster_ids = df_temp[algorithm].sort_values().unique()

        # unique gt cluster_ids for a given claim
        ordered_gt_cluster_ids = df_temp["premiseClusterID_groundTruth"].sort_values().dropna().unique()

        # split premises according to predicted clusters
        splitted_prediction_clusters = split_clusters(df_temp, ordered_predicted_cluster_ids, algorithm)

        # split premises according to gt clusters
        splitted_gt_clusters = split_clusters(df_temp, ordered_gt_cluster_ids, "premiseClusterID_groundTruth")

        # calculate ranking of predicted_clusters
        predicted_ranking = task_b(splitted_prediction_clusters)
        print("predicted_ranking", predicted_ranking)

        # calculate ranking of gt_clusters
        gt_ranking = best_ranking(splitted_gt_clusters)
        print("gt_gain", gt_ranking)

        # calculate ndcg_score for given claim
        ndcg_list.append(ndcg_score(y_score=predicted_ranking, y_true=gt_ranking, k=5))

    # calculate the mean for all 30 query-claim
    print(np.array(ndcg_list).mean())


if __name__ == '__main__':
    main()
