import argparse

import numpy as np

from arclus.evaluation import best_ranking, ndcg_score, split_clusters, task_b
from arclus.utils import load_assignments_with_numeric_relevance


def main():
    """
    Calculates normalized discounted cumulative gain (nDCG) for all queries.
    """
    parser = argparse.ArgumentParser(description='Pre-compute BERT features.')
    parser.add_argument('--algorithm', type=str, default='premiseClusterID_first512Tokens',
                        choices=['premiseClusterID_first512Tokens', 'premiseClusterID_slidingWindow',
                                 'premiseClusterID_sentences'], help='The algorithm which should be evaluated.')
    parser.add_argument('--k', type=int, default=5, choices=[5, 10, ],
                        help='The first k elements in the ranking should be considered')

    args = parser.parse_args()
    algorithm = args.algorithm
    k = args.k
    df = load_assignments_with_numeric_relevance()

    ndcg_list = []
    # ndcg_list_mean = []
    # ndcg_list_max = []
    # ndcg_list_min = []

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

        # calculate ranking of ordered_predicted_clusters
        predicted_ranking = task_b(splitted_prediction_clusters)
        print("predicted_ranking", predicted_ranking)

        # calculate ranking of gt_clusters
        gt_ranking = best_ranking(splitted_gt_clusters)
        print("gt_gain", gt_ranking)

        # calculate ndcg_score for given claim
        ndcg_list.append(ndcg_score(y_score=predicted_ranking, y_true=gt_ranking, k=k))

        # calculate ranking of all possible orders of clusters
        # rankings = task_a(df_temp, splitted_prediction_clusters)
        # scores = []
        # for ranking in rankings:
        #    scores.append(ndcg_score(y_score=ranking, y_true=gt_ranking, k=k))
        # ndcg_list_mean.append(np.array([scores]).mean())
        # ndcg_list_min.append(np.array([scores]).min())
        # ndcg_list_max.append(np.array([scores]).max())

    # calculate all evaluations scores
    # print("task _a mean:", "algorithm:", algorithm, "nDCG@", k, np.array(ndcg_list_mean).mean())
    # print("task _a min:", "algorithm:", algorithm, "nDCG@", k, np.array(ndcg_list_min).mean())
    # print("task _a max:", "algorithm:", algorithm, "nDCG@", k, np.array(ndcg_list_max).mean())
    print("task _b;", "algorithm:", algorithm, "nDCG@", k, np.array(ndcg_list).mean())


if __name__ == '__main__':
    main()
