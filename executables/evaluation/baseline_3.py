import argparse
import logging
from logging import Logger

import numpy as np

from arclus.utils_am import load_bert_model_and_data, inference
from arclus.evaluation import best_ranking, ndcg_score, split_clusters
from arclus.settings import PREP_ASSIGNMENTS_TEST
from arclus.utils import load_assignments_with_numeric_relevance


def main():
    logger: Logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description='Pre-compute BERT features.')
    parser.add_argument('--k', type=int, default=5, choices=[5, 10],
                        help='The first k elements in the ranking should be considered')

    parser.add_argument('--model_path', type=str, default="../../models/d3d4a9c7c23a4b85a20836a754e3aa56",
                        help='Directory where the bert similarity model checkpoint is located')
    parser.add_argument('--pad', type=bool, default=True,
                        help='Should the ranking be padded with 0s until k positions are reached')
    args = parser.parse_args()

    args.data_dir = PREP_ASSIGNMENTS_TEST
    args.task_name = "SIM"
    args.overwrite_cache = True
    args.max_seq_length = 512
    args.model_type = "bert"
    args.batch_size = 128
    k = args.k

    # load bert model and the data
    loader, data, model, guids = load_bert_model_and_data(args)

    # generate logits for all claims-premise pairs
    predictions = inference(args, data, loader, logger, model)

    df = load_assignments_with_numeric_relevance()
    ndcg_list = []
    # iterate over all claims
    start_id = 0
    assert len(predictions) == len(df)
    for id in df["claim_id"].unique():
        # locate all premises which are assigned to the current claim
        premises = df.loc[df["claim_id"] == id]
        premises["similarity"] = predictions[start_id:start_id + len(premises)]
        premises = premises.sort_values(by=['similarity'], ascending=False)
        # generate the ranking (relevance) of the knn premises
        predicted_ranking = premises.relevance.values
        # groundtruth
        ordered_gt_cluster_ids = premises["premiseClusterID_groundTruth"].sort_values().dropna().unique()
        splitted_gt_clusters = split_clusters(premises, ordered_gt_cluster_ids, "premiseClusterID_groundTruth")
        gt_ranking = best_ranking(splitted_gt_clusters)
        gt_ranking.sort(reverse=True)
        # calculate nDCG for the given claim
        ndcg_list.append(ndcg_score(y_score=predicted_ranking, y_true=gt_ranking, k=k, pad=args.pad))
        start_id = start_id + len(premises)
    print("task _b;", "algorithm:", "baseline_3", "nDCG@", k, np.array(ndcg_list).mean())


if __name__ == '__main__':
    main()
