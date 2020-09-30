import argparse
import logging
from logging import Logger

import numpy as np
import pandas as pd

from arclus.utils_am import load_bert_model_and_data, inference
from arclus.evaluation import best_ranking, ndcg_score, split_clusters
from arclus.settings import PREP_CLAIMS_TEST, PREP_ASSIGNMENTS_TEST
from arclus.utils import load_assignments_with_numeric_relevance


def main():
    logger: Logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description='Pre-compute BERT features.')
    parser.add_argument('--k', type=int, default=5, choices=[5, 10],
                        help='The first k elements in the ranking should be considered')

    parser.add_argument('--model_path', type=str, default="../../models/d3d4a9c7c23a4b85a20836a754e3aa56",
                        help='Directory where the bert similarity model checkpoint is located')
    args = parser.parse_args()

    args.data_dir = PREP_ASSIGNMENTS_TEST
    args.task_name = "SIM"
    args.overwrite_cache = True
    args.max_seq_length = 512
    args.model_type = "bert"
    args.batch_size = 128
    k = args.k

    #load bert model and the data
    loader, data, model = load_bert_model_and_data(args)

    #generate logits for all claims-premise pairs
    predictions = inference(args, data, loader, logger, model)

    df_assignments = load_assignments_with_numeric_relevance()
    df_claims = pd.read_csv(PREP_CLAIMS_TEST)

    ndcg_list = []
    # iterate over all claims
    start_id = 0
    assert len(predictions) == len(df_assignments)
    for index, row in df_claims.iterrows():
        # claim id which we use right now
        claim_id = row["claim_id"]

        # locate all premises which are assigned to the current claim
        premises = df_assignments.loc[df_assignments["claim_id"] == claim_id]
        premises["similarity"] = predictions[start_id:start_id + len(premises)]

        premises = premises.sort_values(by=['similarity'], ascending=False)

        # generate the ranking (relevance) of the knn premises
        predicted_ranking = premises.relevance.values

        # groundtruth
        ordered_gt_cluster_ids = premises["premiseClusterID_groundTruth"].sort_values().dropna().unique()
        splitted_gt_clusters = split_clusters(premises, ordered_gt_cluster_ids, "premiseClusterID_groundTruth")
        gt_ranking = best_ranking(splitted_gt_clusters)

        # calculate nDCG for the given claim
        ndcg_list.append(ndcg_score(y_score=predicted_ranking, y_true=gt_ranking, k=k))
        start_id = start_id + len(premises)
    print("task _b;", "algorithm:", "baseline_3", "nDCG@", k, np.array(ndcg_list).mean())


if __name__ == '__main__':
    main()
