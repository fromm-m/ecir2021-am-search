"""Evaluation of baselines."""
import argparse
import logging
import pathlib
from hashlib import sha512

import pandas
import tqdm

from arclus.evaluation import evaluate_ranking_method
from arclus.models import get_baseline_method_by_name
from arclus.similarity import get_similarity_by_name


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', default='results', type=str)
    parser.add_argument('--k', default=[5, 10], type=int)
    parser.add_argument('--force', action='store_true', default=False)
    args = parser.parse_args()

    configs = [
                  # 0. Related Work: Dumani
                  dict(
                      name='dumani',
                      column=column,
                  )
                  for column in ('first512Tokens', 'slidingWindow', 'sentences')
              ] + [
                  # 1. zero_shot_knn
                  dict(
                      name='zero_shot_knn',
                      similarity=similarity,
                  )
                  for similarity in ('l2', 'l1', 'cos')
              ] + [
                  # 2. zero_shot_cluster_knn
                  dict(
                      name='zero_shot_cluster_knn',
                      cluster_ratio=cluster_ratio,
                      cluster_representative=cluster_representative,
                      similarity=similarity,
                  )
                  for cluster_ratio in (0.25, 0.5, 1.0, None)
                  for cluster_representative in ('closest-to-center', 'closest-to-claim')
                  for similarity in ('l2', 'l1', 'cos')
              ] + [
                  # 3. learned_similarity_knn
                  dict(
                      name='learned_similarity_knn',
                      softmax=softmax
                  )
                  for softmax in (False, True)
              ] + [
                  # 4. learned_similarity_cluster_knn
                  dict(
                      name='learned_similarity_cluster_knn',
                      softmax=softmax,
                      cluster_ratio=cluster_ratio,
                  )
                  for softmax in (False, True)
                  for cluster_ratio in (0.25, 0.5, 1.0, None)
              ]

    # ensure output root exists
    output_root = pathlib.Path(args.output_root).expanduser().absolute()
    output_root.mkdir(parents=True, exist_ok=True)

    for config in tqdm.tqdm(configs, unit='configuration', unit_scale=True):
        # determine output path
        hash_digest = sha512(str(sorted(config.items())).encode(encoding='utf8')).hexdigest()[:20]
        output_path = output_root / f'{hash_digest}.tsv'

        # skip existing runs
        if output_path.is_file() and not args.force:
            continue

        # Instantiate similarity if necessary
        if 'similarity' in config.keys():
            config['similarity'] = get_similarity_by_name(name=config['similarity'])

        # Instantiate method
        method = get_baseline_method_by_name(**config)

        # Evaluate method
        result_df = evaluate_ranking_method(method=method, k=args.k)

        # Add configuration
        for key, value in config.items():
            result_df[key] = value

        # Save result
        result_df.to_csv(output_path, index=False, sep='\t')

    # Collate results
    files = list(output_root.glob('*.tsv'))
    if len(files) == 0:
        print('no input files')
        quit(-1)

    adf = []
    for file_path in files:
        df = pandas.read_csv(file_path, sep='\t')
        df['file_path'] = file_path
        adf.append(df)
    df = pandas.concat(
        objs=adf,
        ignore_index=True,
    ).fillna('')
    df['softmax'] = df['softmax'].apply(lambda x: True if x == 1.0 else (False if x == 0.0 else ''))
    mask = df['method'] == 'dumani'
    df.loc[mask, 'similarity'] = df.loc[mask, 'column']
    summary = df.groupby(
        by=[
            'method',
            'similarity',
            'cluster_ratio',
            'cluster_representative',
            'softmax',
            'k',
        ]).agg(dict(mnDCG='mean'))
    summary = summary.unstack()
    print(summary)
    print(summary.to_csv())


if __name__ == '__main__':
    main()
