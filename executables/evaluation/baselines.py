"""Evaluation of baselines."""
import argparse
from datetime import datetime

from arclus.evaluation import evaluate_ranking_method
from arclus.get_similar import get_similarity_by_name
from arclus.models.baselines import get_baseline_method_by_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='zero_shot_knn', type=str)
    parser.add_argument('--k', default=10, type=int)
    parser.add_argument('--similarity', default='l2', type=str)
    args = parser.parse_args()

    # Instantiate method
    method = get_baseline_method_by_name(name=args.method, similarity=get_similarity_by_name(name=args.similarity))
    result_df = evaluate_ranking_method(method=method, k=args.k)
    result_df['method'] = args.method
    result_df['similarity'] = args.similarity
    result_df.to_csv(f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{args.method}_{args.similarity}.csv')
    print(result_df['mnDCG'].mean())


if __name__ == '__main__':
    main()
