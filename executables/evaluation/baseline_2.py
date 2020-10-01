import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from arclus.evaluation import best_ranking, ndcg_score, split_clusters
from arclus.get_similar import LpSimilarity, get_most_similar
from arclus.settings import CLAIMS_TEST_FEATURES, PREMISES_TEST_FEATURES, PREP_CLAIMS_TEST, PREP_PREMISES_TEST
from arclus.utils import load_assignments_with_numeric_relevance


def main():
    """
    Calculates normalized discounted cumulative gain (nDCG) for all queries.
    """
    parser = argparse.ArgumentParser(description='Pre-compute BERT features.')
    parser.add_argument('--k', type=int, default=5, choices=[5, 10, ],
                        help='The first k elements in the ranking should be considered')
    parser.add_argument('--repr', type=str, default="center", choices=["center", "nn_to_claim"],
                        help='How the cluster representative is chosen.')
    parser.add_argument('--pad', type=bool, default=True,
                        help='Should the ranking be padded with 0s until k positions are reached')
    args = parser.parse_args()
    k = args.k
    repr_type = args.repr

    df = load_assignments_with_numeric_relevance()

    claims_representations = torch.load(CLAIMS_TEST_FEATURES)
    premises_representations = torch.load(PREMISES_TEST_FEATURES)

    ndcg_list = []
    # iterate over all claims
    for id in df["claim_id"].unique():
        # locate all premises which are assigned to the current claim
        premises = df.loc[df["claim_id"] == id]
        premises_ids = premises["premise_id"].to_numpy(dtype=str)

        # get the premises representations
        premise_representation_list = [premises_representations[x] for x in premises_ids]
        premises_repr = torch.stack(premise_representation_list)
        assert len(premises_repr) == len(premises_ids)

        # get the claim representation
        claim_repr = claims_representations[id].unsqueeze(dim=0)

        n_clusters = round(len(premises_repr) / 2)
        n_clusters = max([n_clusters, k])
        # cluster all premises, n_clusters can be chosen
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(premises_repr)
        prepare_centers = torch.from_numpy(kmeans.cluster_centers_)

        # choose representative of each cluster
        if repr_type == "center":
            # choose nearest to cluster centers as representative for each cluster
            repr = [
                get_most_similar(torch.reshape(center, (1, len(center))), premises_repr.double(), 1,
                                 LpSimilarity()) for center in prepare_centers]
        else:
            premises_per_cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

            # choose representative, here: nearest to claim
            repr = [
                get_most_similar(claim_repr, torch.from_numpy(premises_representations[premises_per_cluster[i]]), 1,
                                 LpSimilarity()) for i in range(kmeans.n_clusters)]

        # format representatives
        representatives = torch.cat([x[0].reshape(-1, x[0].shape[-1]) for x in repr])
        repr_ind = torch.cat([x[1].reshape(-1) for x in repr])

        # choose the nearest premises to claim representation among cluster representatives
        k_premises, k_indices = get_most_similar(
            claims=claim_repr.float(),
            premises=representatives.float(),
            k=k,
            similarity=LpSimilarity()
        )
        k_indices = torch.index_select(repr_ind, 0, k_indices.reshape(-1))

        # select the premises by index
        predicted_ranking = df.iloc[k_indices.squeeze().numpy()].relevance.to_numpy(dtype=int)

        # groundtruth
        ordered_gt_cluster_ids = premises["premiseClusterID_groundTruth"].sort_values().dropna().unique()
        splitted_gt_clusters = split_clusters(premises, ordered_gt_cluster_ids, "premiseClusterID_groundTruth")
        gt_ranking = best_ranking(splitted_gt_clusters)
        gt_ranking.sort(reverse=True)

        # calculate nDCG for the given claim
        ndcg_list.append(ndcg_score(y_score=predicted_ranking, y_true=gt_ranking, k=k, pad=args.pad))

    print("task _b;", "algorithm:", "baseline_2", "nDCG@", k, np.array(ndcg_list).mean())


if __name__ == '__main__':
    main()
