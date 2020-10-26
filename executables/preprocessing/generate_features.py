"""Precompute BERT features."""
import argparse
import logging

import pandas

from arclus.settings import (CLAIMS_LENGTH, CLAIMS_TEST_FEATURES, PREMISES_FEATURES,
                             PREP_ASSIGNMENTS, PREP_CLAIMS, PREP_CLAIMS_TEST, PREP_PREMISES_ENERGY, PREP_PREMISES,
                             PREMISES_LENGTH)
from arclus.text_encoding import encode_and_save_texts
from arclus.utils import concat_premise_claims


# premises bert server = 275
# claims bert server = 25


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Pre-compute BERT features.')
    parser.add_argument('--mode', default='pair', choices=['pair', 'claims', 'premises'], help='The feature mode.')
    args = parser.parse_args()
    size = 325000

    if args.mode == 'pair':
        premises_df = pandas.read_csv(PREP_PREMISES_ENERGY, delimiter=",")
        claims_df = pandas.read_csv(PREP_CLAIMS, delimiter=",")
        ass_df = pandas.read_csv(PREP_ASSIGNMENTS, delimiter=",")
        texts = concat_premise_claims(
            premises_df=premises_df,
            claims_df=claims_df,
            assignment_df=ass_df,
            max_premise_words=PREMISES_LENGTH,
            max_claim_words=CLAIMS_LENGTH,
        )
    elif args.mode == 'claims':
        df = pandas.read_csv(PREP_CLAIMS_TEST)
        texts = df['claim_text']
        keys = df['claim_id']
        output_path = CLAIMS_TEST_FEATURES
    elif args.mode == 'premises':
        df = pandas.read_csv(PREP_PREMISES)
        print(len(df))
        df = df.head(size)
        texts = df['premise_text']
        keys = df['premise_id']
        output_path = PREMISES_FEATURES
    else:
        raise ValueError(f'Invalid mode={args.mode}')

    # Delegate to library method
    assert (len(keys) == len(texts))
    encode_and_save_texts(texts=texts, output_path=output_path, keys=keys)


if __name__ == '__main__':
    main()
