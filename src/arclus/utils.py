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
