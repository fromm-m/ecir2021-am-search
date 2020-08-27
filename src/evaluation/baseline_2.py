import pandas as pd
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch

from arclus.settings import PREP_ASSIGNMENTS_TEST, PREP_CLAIMS_TEST, PREP_PREMISES_TEST, CLAIMS_TEST_FEATURES, \
    PREMISES_TEST_FEATURES
from evaluation.reproduce_scores import split_clusters, ndcg_score, best_ranking
from arclus.get_similar import get_most_similar, LpSimilarity


def main():
    """
    Calculates normalized discounted cumulative gain (nDCG) for all queries.
    """
    parser = argparse.ArgumentParser(description='Pre-compute BERT features.')
    parser.add_argument('--k', type=int, default=5, choices=[5, 10, ],
                        help='The first k elements in the ranking should be considered')

    parser.add_argument('--repr', type=str, default="center", choices=["center", "nn_to_claim"],
                        help='How the cluster representative is chosen.')

    args = parser.parse_args()
    k = args.k
    repr_type = args.repr

    df_assignments = pd.read_csv(PREP_ASSIGNMENTS_TEST, sep=";")
    df_claims = pd.read_csv(PREP_CLAIMS_TEST)
    df_premises = pd.read_csv(PREP_PREMISES_TEST)

    # set the relevance to the according value (cf. paper)
    df_assignments['relevance'].loc[(df_assignments['relevance'] == "notRelevant")] = 0
    df_assignments['relevance'].loc[(df_assignments['relevance'] == "yesRelevant")] = 1
    df_assignments['relevance'].loc[(df_assignments['relevance'] == "yesVeryRelevant")] = 2

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

        n_clusters = round(len(filtered_premise_representation) / 2)
        n_clusters = max([n_clusters, k])

        # cluster all premises, n_clusters can be chosen
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(filtered_premise_representation)
        prepare_centers = torch.from_numpy(kmeans.cluster_centers_)

        # choose representative of each cluster
        if repr_type == "center":
            # choose nearest to cluster centers as representative for each cluster
            repr = [
                get_most_similar(torch.reshape(center, (1, len(center))), torch.from_numpy(premises_representations), 1,
                                 LpSimilarity()) for center in prepare_centers]
        else:
            premises_per_cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

            # choose representative, here: nearest to claim
            repr = [
                get_most_similar(claims_torch, torch.from_numpy(premises_representations[premises_per_cluster[i]]), 1,
                                 LpSimilarity()) for i in range(kmeans.n_clusters)]

        # format representatives
        representatives = torch.cat([x[0].reshape(-1, x[0].shape[-1]) for x in repr])
        repr_ind = torch.cat([x[1].reshape(-1) for x in repr])

        # choose the nearest premises to claim representation among cluster representatives
        k_premises, k_indices = get_most_similar(
            claims=claims_torch,
            premises=representatives,
            k=k,
            similarity=LpSimilarity()
        )
        k_indices = torch.index_select(repr_ind, 0, k_indices.reshape(-1))

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

    print("task _b;", "algorithm:", "baseline_2", "nDCG@", k, np.array(ndcg_list).mean())


if __name__ == '__main__':
    main()
