import argparse

import numpy as np
import pandas as pd
import torch

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

    args = parser.parse_args()
    k = args.k
    df_claims = pd.read_csv(PREP_CLAIMS_TEST)
    df_premises = pd.read_csv(PREP_PREMISES_TEST)

    df_assignments = load_assignments_with_numeric_relevance()

    claims_representations = np.load(CLAIMS_TEST_FEATURES)
    premises_representations = np.load(PREMISES_TEST_FEATURES)

    ndcg_list = []
    # iterate over all claims
    for index, row in df_claims.iterrows():
        # get the claim representation
        claim_repr = claims_representations[index]

        # claim id which we use right now
        claim_id = row["claim_id"]

        # locate all premises which are assigned to the current claim
        premises = df_assignments.loc[df_assignments["queryClaimID"] == claim_id]
        premises_ids = premises["resultClaimsPremiseID"].values
        mask = df_premises['premise_id'].isin(premises_ids)
        premise_index = df_premises.index[mask]
        assert len(premise_index) == len(premises_ids)

        # get the premises representations
        filtered_premise_representation = premises_representations[premise_index]
        assert len(filtered_premise_representation) == len(premise_index)

        # convert the claims and premises to tensors
        claims_torch = torch.from_numpy(claim_repr)
        claims_torch = torch.reshape(claims_torch, (1, len(claims_torch)))
        premises_torch = torch.from_numpy(filtered_premise_representation)

        # find knn premises given a claim
        k_premises, k_indices = get_most_similar(
            claims=claims_torch,
            premises=premises_torch,
            k=k,
            similarity=LpSimilarity()
        )

        # select the premises by index
        k_premise_ids = df_premises.iloc[k_indices.reshape(k)].premise_id.values
        mask_2 = df_assignments['resultClaimsPremiseID'].isin(k_premise_ids)
        k_premise_df = df_assignments[mask_2]

        # generate the ranking (relevance) of the knn premises
        predicted_ranking = k_premise_df.relevance.values

        # groundtruth
        ordered_gt_cluster_ids = premises["premiseClusterID_groundTruth"].sort_values().dropna().unique()
        splitted_gt_clusters = split_clusters(premises, ordered_gt_cluster_ids, "premiseClusterID_groundTruth")
        gt_ranking = best_ranking(splitted_gt_clusters)

        # calculate nDCG for the given claim
        ndcg_list.append(ndcg_score(y_score=predicted_ranking, y_true=gt_ranking, k=k))

    print("task _b;", "algorithm:", "baseline_1", "nDCG@", k, np.array(ndcg_list).mean())


if __name__ == '__main__':
    main()
