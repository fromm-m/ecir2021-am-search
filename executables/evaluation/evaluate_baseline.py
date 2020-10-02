"""Evaluation of baselines."""
import argparse
import pathlib
from datetime import datetime

from arclus.evaluation import evaluate_ranking_method
from arclus.models.baselines import get_baseline_method_by_name
from arclus.similarity import get_similarity_by_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', default='results', type=str)
    parser.add_argument('--method', default='zero_shot_knn', type=str)
    parser.add_argument('--k', default=[5, 10], type=int)
    parser.add_argument('--similarity', default='l2', type=str)
    parser.add_argument('--cluster_ratio', default=0.5, type=float)
    args = parser.parse_args()

    output_root = pathlib.Path(args.output_root).expanduser().absolute()
    print(f'Output to {str(output_root)}')
    output_root.mkdir(parents=True, exist_ok=True)

    # Instantiate method
    method = get_baseline_method_by_name(
        name=args.method,
        similarity=get_similarity_by_name(name=args.similarity),
        cluster_ratio=args.cluster_ratio,
    )
    result_df = evaluate_ranking_method(method=method, k=args.k)
    result_df['method'] = args.method
    result_df['similarity'] = args.similarity
    result_df.to_csv(output_root / f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{args.method}.csv', index=False)


if __name__ == '__main__':
    main()
