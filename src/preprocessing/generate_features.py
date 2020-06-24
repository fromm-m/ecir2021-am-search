"""
Precompute BERT embeddings for claims and premises.
"""
import logging

import numpy
import pandas
from bert_serving.client import BertClient

from arclus.settings import PREP_CLAIMS, PREP_PREMISES, PREP_ASSIGNMENTS, OUTPUT_FEATURES, CLAIMS_LENGTH, \
    PREMISES_LENGTH


def truncate(x: str) -> str:
    a = x.split()[0:CLAIMS_LENGTH]
    return " ".join(a)


def main():
    logging.basicConfig(level=logging.INFO)

    # Connect to BERT-as-a-service
    client = BertClient()

    premises_df = pandas.read_csv(PREP_PREMISES, delimiter=",")
    claims_df = pandas.read_csv(PREP_CLAIMS, delimiter=",")
    ass_df = pandas.read_csv(PREP_ASSIGNMENTS, delimiter=",")
    ass_extended = pandas.merge(ass_df, premises_df, how='inner', on="premise_id")
    ass_extended = pandas.merge(ass_extended, claims_df, how='inner', on="claim_id")
    ass_extended = ass_extended.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y', "Unnamed: 0"])

    premises = ass_extended['premise_text']
    claims = ass_extended['claim_text']
    premises = premises.apply(truncate, args=PREMISES_LENGTH)
    claims = claims.apply(truncate, args=CLAIMS_LENGTH)

    pair = premises + ' ||| ' + claims
    pair = list(pair.values.flatten())
    embeddings = client.encode(texts=pair)
    numpy.save(OUTPUT_FEATURES, embeddings)


if __name__ == '__main__':
    main()
