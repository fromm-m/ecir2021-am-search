"""Evaluation of baselines."""
import argparse
import logging
import pathlib
from hashlib import sha512

from arclus.evaluation import evaluate_ranking_method
from arclus.models.baselines import get_baseline_method_by_name
from arclus.similarity import get_similarity_by_name
from arclus.utils import argparse_bool


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', default='results', type=str)
    parser.add_argument('--method', default='zero_shot_knn', type=str)
    parser.add_argument('--k', default=[5, 10], type=int)
    parser.add_argument('--similarity', default=None, type=str)
    parser.add_argument('--softmax', default=None, type=argparse_bool)
    parser.add_argument('--cluster_ratio', default=None, type=float)
    parser.add_argument('--cluster_representative', default=None, type=str, choices=['closest-to-center', 'closest-to-claim'])
    parser.add_argument('--column', default=None, type=str, choices=["first512Tokens", "slidingWindow", "sentences"])
    parser.add_argument('--force', action='store_true', default=False)
    args = parser.parse_args()
    config = dict(
        method=args.method,
        similarity=args.similarity,
        cluster_ratio=args.cluster_ratio,
        cluster_representative=args.cluster_representative,
        softmax=args.softmax,
        column=args.column,
    )

    output_root = pathlib.Path(args.output_root).expanduser().absolute()
    output_root.mkdir(parents=True, exist_ok=True)

    hash_digest = sha512(str(sorted(config.items())).encode(encoding='utf8')).hexdigest()[:20]
    output_path = output_root / f'{hash_digest}.tsv'
    print('Output to', output_path)
    if output_path.is_file() and not args.force:
        quit(0)

    # Instantiate method
    method = get_baseline_method_by_name(
        name=args.method,
        similarity=get_similarity_by_name(name=args.similarity) if args.similarity is not None else None,
        cluster_ratio=args.cluster_ratio,
        cluster_representative=args.cluster_representative,
        softmax=args.softmax,
        column=args.column,
    )
    result_df = evaluate_ranking_method(method=method, k=args.k)
    for key, value in config.items():
        result_df[key] = value
    result_df.to_csv(output_path, index=False, sep='\t')


if __name__ == '__main__':
    main()
