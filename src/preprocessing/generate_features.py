"""
Precompute BERT embeddings for claims and premises.
"""
import logging

import numpy
import pandas
from bert_serving.client import BertClient

from arclus.settings import PREP_CLAIMS, PREP_PREMISES


def main():
    logging.basicConfig(level=logging.INFO)

    # Connect to BERT-as-a-service
    client = BertClient()

    for source_path, text_col_name in zip(
        (PREP_CLAIMS, PREP_PREMISES),
        ('premise_text', 'claim_text'),
    ):
        output_path = source_path.parent / source_path.name.replace('.csv', '.npy')
        if output_path.is_file():
            logging.info(f'Skipping existing file: {output_path}')
            continue
        df = pandas.read_csv(source_path, delimiter=',')
        embeddings = client.encode(
            texts=df[text_col_name].tolist(),
        )
        numpy.save(output_path, embeddings)


if __name__ == '__main__':
    main()
