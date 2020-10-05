"""Ranking methods."""
import inspect
from typing import Any

from .base import RankingMethod
from .baselines import LearnedSimilarityClusterKNN, LearnedSimilarityKNN, ZeroShotClusterKNN, ZeroShotKNN  # noqa: F401
from .dumani import Dumani  # noqa: F401
from ..utils import get_subclass_by_name


def name_normalizer(name: str) -> str:
    return name.lower().replace('_', '')


def get_baseline_method_by_name(
    name: str,
    **kwargs: Any,
) -> RankingMethod:
    cls = get_subclass_by_name(base_class=RankingMethod, name=name, normalizer=name_normalizer)
    real_kwargs = dict()
    for key, value in kwargs.items():
        signature = inspect.signature(cls.__init__)
        if key in signature.parameters:
            real_kwargs[key] = value
        else:
            print(f"Unused argument {key}={value}")
    return cls(**real_kwargs)
