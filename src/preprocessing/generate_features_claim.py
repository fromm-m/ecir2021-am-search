"""
Precompute BERT embeddings for claims.
"""
import logging

import numpy
import pandas
from bert_serving.client import BertClient

from arclus.settings import PREP_CLAIMS_TEST, CLAIMS_TEST_FEATURES, CLAIMS_LENGTH

def main():
    logging.basicConfig(level=logging.INFO)

    # Connect to BERT-as-a-service
    client = BertClient()

    claims_df = pandas.read_csv(PREP_CLAIMS_TEST)
    claims = claims_df['claim_text']
    claims = list(claims.values.flatten())
    claims_embeddings = client.encode(texts=claims)
    numpy.save(CLAIMS_TEST_FEATURES, claims_embeddings)


if __name__ == '__main__':
    main()
