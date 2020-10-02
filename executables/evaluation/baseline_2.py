import argparse

import numpy as np
import torch

from arclus.evaluation import best_ranking, evaluate_premises, ndcg_score, split_clusters
from arclus.settings import CLAIMS_TEST_FEATURES, PREMISES_TEST_FEATURES
from arclus.similarity import LpSimilarity, get_most_similar
from arclus.utils import load_assignments_with_numeric_relevance
from arclus.utils_clustering import clustering


def main():
    """
    Calculates normalized discounted cumulative gain (nDCG) for all queries with zero shot bert, clustering and knn
    """
    parser = argparse.ArgumentParser(description='Pre-compute BERT features.')
    parser.add_argument('--k', type=int, default=5, choices=[5, 10, ],
                        help='The first k elements in the ranking should be considered')
    parser.add_argument('--repr', type=str, default="center", choices=["center", "nn_to_claim"],
                        help='How the cluster representative is chosen.')
    parser.add_argument('--pad', type=bool, default=True,
                        help='Should the ranking be padded with 0s until k positions are reached')
    args = parser.parse_args()

    df = load_assignments_with_numeric_relevance()
    claims_representations = torch.load(CLAIMS_TEST_FEATURES)
    premises_representations = torch.load(PREMISES_TEST_FEATURES)

    ndcg_list = []
    # iterate over all claims
    for claim_id in df["claim_id"].unique():
        # locate all premises which are assigned to the current claim
        premises = df.loc[df["claim_id"] == claim_id]
        premises_ids = premises["premise_id"].to_numpy(dtype=str)

        # get the premises representations
        premise_representations = torch.stack([premises_representations[x] for x in premises_ids])
        assert len(premise_representations) == len(premises_ids)

        # get the claim representation
        claim_representation = claims_representations[claim_id].unsqueeze(dim=0)

        # do clustering
        repr_ind, representatives = clustering(args, claim_representation, premise_representations)

        # choose the nearest premises to claim representation among cluster representatives
        k_premises, k_indices = get_most_similar(
            claims=claim_representation.float(),
            premises=representatives.float(),
            k=args.k,
            similarity=LpSimilarity()
        )
        k_indices = torch.index_select(repr_ind, 0, k_indices.reshape(-1))

        # select the premises by index
        knn_premises = premises.iloc[k_indices.squeeze().numpy()]
        predicted_ranking = evaluate_premises(knn_premises)

        # groundtruth
        ordered_gt_cluster_ids = premises["premiseClusterID_groundTruth"].sort_values().dropna().unique()
        splitted_gt_clusters = split_clusters(premises, ordered_gt_cluster_ids, "premiseClusterID_groundTruth")
        gt_ranking = best_ranking(splitted_gt_clusters)
        gt_ranking.sort(reverse=True)

        # calculate nDCG for the given claim
        ndcg_list.append(ndcg_score(y_score=predicted_ranking, y_true=gt_ranking, k=args.k, pad=args.pad))

    print("task _b;", "algorithm:", "baseline_2", "nDCG@", args.k, np.array(ndcg_list).mean())


if __name__ == '__main__':
    main()
