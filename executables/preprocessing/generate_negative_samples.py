"""Script to generate all negative samples for training."""
import argparse

import pandas as pd

from arclus.settings import NEGATIVE_SAMPLES, POSITIVE_SAMPLES, PREP_ASSIGNMENTS, PREP_CLAIMS


def main(*keywords: str):
    assignments = pd.read_csv(PREP_ASSIGNMENTS)
    print('loaded', assignments.shape[0], 'positive samples.')

    claims = pd.read_csv(PREP_CLAIMS)
    print('loaded', claims.shape[0], 'claims.')

    # filter claims by keywords
    claims: pd.DataFrame = claims.loc[claims["claim_text"].str.contains('|'.join(f'({key})' for key in keywords), regex=True), ["claim_id"]]
    print('Kept', claims.shape[0], 'claims.')

    pairs = claims.merge(right=assignments, how="inner", on="claim_id")

    # save positive pairs
    pairs.to_csv(POSITIVE_SAMPLES)
    print('wrote', pairs.shape[0], 'positive samples.')

    # save all negative pairs
    all_premises = set(pairs["premise_id"].unique().tolist())
    pool = dict()
    for claim_id, group in pairs.groupby(by='claim_id'):
        pool[claim_id] = all_premises.difference(group["premise_id"].unique().tolist())
    print('In total', sum(map(len, pool.values())), 'negative samples.')

    all_negatives = {
        (pid, cid)
        for pid, cids in pool.items()
        for cid in cids
    }
    negative_pairs = pd.DataFrame(data=list(all_negatives), columns=["premise_id", "claim_id"])
    negative_pairs.to_csv(NEGATIVE_SAMPLES)
    print('wrote', negative_pairs.shape[0], 'negative samples.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyword', nargs='+', type=str)
    args = parser.parse_args()

    keywords = 'energy, gas, fuel, solar, global warming, natural source, nuclear, renewable, climate change, opec, facturing, fusion, pollution, biofuels, wind'.split(sep=', ')

    main(*keywords)
