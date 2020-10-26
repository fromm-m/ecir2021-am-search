"""Evaluation of baselines."""
import argparse
import copy
import logging
import pathlib
import re
from hashlib import sha512
from operator import itemgetter

import mlflow
import pandas
import tqdm

from arclus.models import get_baseline_method_by_name
from arclus.models.learned_similarity import PremiseRepresentationEnum
from arclus.pipeline import evaluate_ranking_method
from arclus.settings import DATA_ROOT
from arclus.similarity import get_similarity_by_name
from arclus.utils import flatten_dict

logging.basicConfig(level=logging.ERROR)


def main():
    mlflow.set_tracking_uri("http://mlflow.dbs.ifi.lmu.de:5000")
    mlflow.set_experiment(experiment_name="argument_clustering")
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', default='results', type=str)
    parser.add_argument('--k', default=[5, 10], type=int)
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--filter', default=None, type=str, required=False, nargs='*')
    args = parser.parse_args()
    similarities = DATA_ROOT / 'similarities'
    model_path = '/nfs/data3/fromm/argument_clustering/models/c26817b78afc4e95ab86793d7390081b'
    PREMISE_REPRESENTATIONS = [
        PremiseRepresentationEnum.learned_similarity_last_layer,
        PremiseRepresentationEnum.learned_similarity_claim_similarities,
    ]
    CLUSTER_RATIOS = (
        0.25,
        0.5,
        1.0,
        None,
    )
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
                      cluster_ratios=CLUSTER_RATIOS,
                      cluster_representative=cluster_representative,
                      similarity=similarity,
                  )
                  for cluster_representative in ('closest-to-center', 'closest-to-claim')
                  for similarity in ('l2', 'l1', 'cos')
              ] + [
                  # 3. learned_similarity_knn
                  dict(
                      name='learned_similarity_knn',
                      softmax=softmax,
                      similarities_dir=similarities,
                      model_path=model_path,
                  )
                  for softmax in (False, True)
              ] + [
                  # 4. learned_similarity_cluster_knn
                  dict(
                      name='learned_similarity_cluster_knn',
                      softmax=softmax,
                      cluster_ratios=CLUSTER_RATIOS,
                      similarities_dir=similarities,
                      model_path=model_path,

                  )
                  for softmax in (False, True)
              ] + [
                  # 5. learned_similarity_cluster_knn
                  dict(
                      name='learned_similarity_matrix_cluster_knn',
                      softmax=softmax,
                      cluster_ratio=cluster_ratio,
                      similarities_dir=similarities,
                      model_path=model_path,
                  )
                  for softmax in (False, True)
                  for cluster_ratio in CLUSTER_RATIOS
              ] + [
                  # 6. Coreset
                  dict(
                      name="coreset",
                      premise_premise_similarity=similarity,
                      similarities_dir=similarities,
                      model_path=model_path,
                      debug=True,
                      fill_to_k=fill_to_k,
                      premise_representation=premise_representation,
                  )
                  for similarity in ('l2', 'l1', 'cos')
                  for fill_to_k in (False, True)
                  for premise_representation in PREMISE_REPRESENTATIONS
              ] + [
                  # Biased Coreset
                  dict(
                      name="biased_coreset",
                      premise_premise_similarity=similarity,
                      similarities_dir=similarities,
                      model_path=model_path,
                      debug=True,
                      resolution=50,
                      premise_representation=premise_representation,
                  )
                  for similarity in ('l2', 'l1', 'cos')
                  for premise_representation in PREMISE_REPRESENTATIONS
              ]

    # filter configurations based on keywords
    if args.filter is not None:
        for one_filter in args.filter:
            k, v = one_filter.split(":", maxsplit=1)
            configs = [
                config
                for config in configs
                if all(
                    not re.match(k, kk) or re.match(v, str(vv))
                    for kk, vv in config.items()
                )
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

        with mlflow.start_run():
            log_config = copy.deepcopy(config)
            log_config["k"] = args.k
            mlflow.log_params(params=flatten_dict(log_config))

            # Instantiate similarity if necessary
            real_config = config.copy()
            for key in ("similarity", "premise_premise_similarity"):
                if key in config.keys():
                    real_config[key] = get_similarity_by_name(name=config[key])

            # Instantiate method
            method = get_baseline_method_by_name(**real_config)

            # Evaluate method
            result_df = evaluate_ranking_method(method=method, k=args.k)
            agg = result_df.groupby(by="k").agg(dict(mnDCG="mean"))
            mlflow.log_metrics(metrics={
                f"mnDCG_at_{k}": mndcg
                for k, mndcg in zip(agg.index.tolist(), map(itemgetter(0), agg.values.tolist()))
            })

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
    mask = df['name'] == 'dumani'
    df.loc[mask, 'similarity'] = df.loc[mask, 'column']
    summary = df.groupby(
        by=[
            'name',
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
