"""Script to generate all negative samples for training."""
import argparse

import pandas as pd

from arclus.settings import NEGATIVE_SAMPLES, POSITIVE_SAMPLES, PREP_ASSIGNMENTS, PREP_CLAIMS


def main(*keywords: str):
    assignments = pd.read_csv(PREP_ASSIGNMENTS)
    claims = pd.read_csv(PREP_CLAIMS)

    # filter claims by keywords
    claims: pd.DataFrame = claims.loc[claims["claim_text"].str.contains('|'.join(f'({key})' for key in keywords), regex=True), ["claim_id"]]
    pairs = claims.merge(right=assignments, how="inner", on="claim_id")

    # save positive pairs
    pairs.to_csv(POSITIVE_SAMPLES)

    # save all negative pairs
    all_positives = set(zip(pairs["premise_id"].tolist(), pairs["claim_id"].tolist()))
    all_premises, all_claims = set(pairs[col].unique().tolist() for col in ["premise_id", "claim_id"])
    all_negatives = {
        (pid, cid)
        for pid in all_premises
        for cid in all_claims
        if (pid, cid) not in all_positives
    }
    negative_pairs = pd.DataFrame(data=list(all_negatives), columns=["premise_id", "claim_id"])
    negative_pairs.to_csv(NEGATIVE_SAMPLES)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyword', nargs='+', type=str)
    args = parser.parse_args()

    main(keywords=args.keyword)
