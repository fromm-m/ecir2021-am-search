"""Precompute BERT features."""
import argparse
import logging

import pandas

from arclus.settings import (CLAIMS_LENGTH, CLAIMS_TEST_FEATURES, OUTPUT_FEATURES, PREMISES_TEST_FEATURES, PREP_ASSIGNMENTS, PREP_CLAIMS, PREP_CLAIMS_TEST, PREP_PREMISES, PREP_PREMISES_TEST, PREMISES_LENGTH)
from arclus.text_encoding import encode_and_save_texts
from arclus.utils import concat_premise_claims


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Pre-compute BERT features.')
    parser.add_argument('--mode', default='pair', choices=['pair', 'claims', 'premises'], help='The feature mode.')
    args = parser.parse_args()

    if args.mode == 'pair':
        premises_df = pandas.read_csv(PREP_PREMISES, delimiter=",")
        claims_df = pandas.read_csv(PREP_CLAIMS, delimiter=",")
        ass_df = pandas.read_csv(PREP_ASSIGNMENTS, delimiter=",")
        texts = concat_premise_claims(
            premises_df=premises_df,
            claims_df=claims_df,
            assignment_df=ass_df,
            max_premise_words=PREMISES_LENGTH,
            max_claim_words=CLAIMS_LENGTH,
        )
        output_path = OUTPUT_FEATURES
    elif args.mode == 'claims':
        texts = pandas.read_csv(PREP_CLAIMS_TEST)['claim_text']
        output_path = CLAIMS_TEST_FEATURES
    elif args.mode == 'premises':
        texts = pandas.read_csv(PREP_PREMISES_TEST)['premise_text']
        output_path = PREMISES_TEST_FEATURES
    else:
        raise ValueError(f'Invalid mode={args.mode}')

    # Delegate to library method
    encode_and_save_texts(texts=texts, output_path=output_path)


if __name__ == '__main__':
    main()
