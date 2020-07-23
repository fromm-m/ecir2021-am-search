"""
Precompute BERT embeddings for premises.
"""
import logging

import numpy
import pandas
from bert_serving.client import BertClient

from arclus.settings import PREP_PREMISES_TEST, PREMISES_TEST_FEATURES, PREMISES_LENGTH


def main():
    logging.basicConfig(level=logging.INFO)

    # Connect to BERT-as-a-service
    client = BertClient()

    premises_df = pandas.read_csv(PREP_PREMISES_TEST)
    premises = premises_df['premise_text']
    premises = list(premises.values.flatten())
    print("Premises", len(premises))
    premises_embeddings = client.encode(texts=premises)
    numpy.save(PREMISES_TEST_FEATURES, premises_embeddings)


if __name__ == '__main__':
    main()
