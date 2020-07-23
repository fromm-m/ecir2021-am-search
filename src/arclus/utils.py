"""
Some utilities functions.
"""
import random

import numpy as np

import torch


def set_random_seed(seed: int):
    """Set the random seed on numpy, torch, and python."""
    return (
        np.random.seed(seed=seed),
        torch.manual_seed(seed=seed),
        random.seed(seed),
    )


def flatten_dict(d):
    """
    Function to transform a nested dictionary to a flattened dot notation dictionary.

    :param d: Dict
        The dictionary to flatten.

    :return: Dict
        The flattened dictionary.
    """

    def expand(key, value):
        if isinstance(value, dict):
            return [(key + '.' + k, v) for k, v in flatten_dict(value).items()]
        else:
            return [(key, value)]

    items = [item for k, v in d.items() for item in expand(k, v)]
    return dict(items)
