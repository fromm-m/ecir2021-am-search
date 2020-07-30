"""Generate BERT features."""
import logging
import pathlib
from typing import List, Union

import numpy
import pandas
from bert_serving.client import BertClient


def encode_and_save_texts(
    texts: Union[pandas.Series, List[str], numpy.ndarray],
    output_path: Union[pathlib.Path, str],
) -> None:
    """
    Generate BERT features for texts using BERT-as-a-service.

    :param texts:
        The texts.
    :param output_path:
        The output path where to save the embeddings.
    """
    # Connect to BERT-as-a-service
    logging.info('Connecting to BERT-as-a-service')
    client = BertClient()

    # Normalize texts
    if isinstance(texts, (pandas.Series, pandas.DataFrame)):
        texts = texts.values
    if isinstance(texts, numpy.ndarray):
        texts = list(texts.flatten())
    assert isinstance(texts, list)
    assert all(isinstance(text, str) for text in texts)

    # encode
    embeddings = client.encode(texts=texts)

    # Normalize path
    if isinstance(output_path, str):
        output_path = pathlib.Path(output_path)

    # ensure parent directory exists
    output_path.parent.mkdir(parents=True, exists_ok=True)

    # save to file
    numpy.save(output_path, embeddings)
