import argparse

import logging
from logging import Logger
import numpy as np
import torch
from sklearn.cluster import KMeans

from arclus.utils_am import load_bert_model_and_data, inference
from arclus.utils_clustering import clustering
from arclus.evaluation import best_ranking, ndcg_score, split_clusters, evaluate_premises
from arclus.get_similar import LpSimilarity, get_most_similar
from arclus.settings import CLAIMS_TEST_FEATURES, PREMISES_TEST_FEATURES, PREP_ASSIGNMENTS_TEST
from arclus.utils import load_assignments_with_numeric_relevance


def main():
    logger: Logger = logging.getLogger(__name__)

    """
    Calculates normalized discounted cumulative gain (nDCG) for all queries.
    """
    parser = argparse.ArgumentParser(description='Pre-compute BERT features.')
    parser.add_argument('--k', type=int, default=5, choices=[5, 10, ],
                        help='The first k elements in the ranking should be considered')

    parser.add_argument('--repr', type=str, default="center", choices=["center", "nn_to_claim"],
                        help='How the cluster representative is chosen.')

    parser.add_argument('--model_path', type=str, default="../../models/d3d4a9c7c23a4b85a20836a754e3aa56",
                        help='Directory where the bert similarity model checkpoint is located')
    parser.add_argument('--pad', type=bool, default=True,
                        help='Should the ranking be padded with 0s until k positions are reached')
    args = parser.parse_args()

    # define parameters
    args.data_dir = PREP_ASSIGNMENTS_TEST
    args.task_name = "SIM"
    args.overwrite_cache = True
    args.max_seq_length = 512
    args.model_type = "bert"
    args.batch_size = 128
    k = args.k
    repr_type = args.repr

    # load bert model and the data
    loader, data, model, guids = load_bert_model_and_data(args)

    # generate logits for all claims-premise pairs
    predictions = inference(args, data, loader, logger, model)

    # combine premise_ids and similarity in a dictionary
    d = dict(zip(guids, predictions))

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
        premise_representations = torch.stack([premises_representations[x] for x in premises_ids])
        assert len(premise_representations) == len(premises_ids)

        # get the claim representation
        claim_representation = claims_representations[id].unsqueeze(dim=0)

        # do clustering
        repr_ind, representatives = clustering(args, claim_representation, premise_representations)

        # choose the nearest premises to claim representation among cluster representatives
        k_premises, k_indices = get_most_similar(
            claims=claim_representation.float(),
            premises=representatives.float(),
            k=k,
            similarity=LpSimilarity()
        )
        k_indices = torch.index_select(repr_ind, 0, k_indices.reshape(-1))

        k_premise_df = premises.iloc[k_indices.squeeze().numpy()]

        k_premise_df["similarity"] = k_premise_df['premise_id'].map(d)
        k_premise_df = k_premise_df.sort_values(by=['similarity'], ascending=False)

        # generate the ranking (relevance) of the knn premises
        predicted_ranking = evaluate_premises(k_premise_df)

        # groundtruth
        ordered_gt_cluster_ids = premises["premiseClusterID_groundTruth"].sort_values().dropna().unique()
        splitted_gt_clusters = split_clusters(premises, ordered_gt_cluster_ids, "premiseClusterID_groundTruth")
        gt_ranking = best_ranking(splitted_gt_clusters)
        gt_ranking.sort(reverse=True)

        # calculate nDCG for the given claim
        ndcg_list.append(ndcg_score(y_score=predicted_ranking, y_true=gt_ranking, k=k, pad=args.pad))

    print("task _b;", "algorithm:", "baseline_4", "nDCG@", k, np.array(ndcg_list).mean())


if __name__ == '__main__':
    main()
