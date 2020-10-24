"""Ranking methods."""
import inspect
from typing import Any

from .base import RankingMethod
from .dumani import Dumani  # noqa: F401
from .learned_similarity import LearnedSimilarityBasedMethod, LearnedSimilarityClusterKNN  # noqa: F401
from .zero_shot import ZeroShotClusterKNN, ZeroShotKNN  # noqa: F401
from ..utils import get_subclass_by_name


def name_normalizer(name: str) -> str:
    """Normalize the name of a method."""
    return name.lower().replace('_', '')


def get_baseline_method_by_name(
    name: str,
    **kwargs: Any,
) -> RankingMethod:
    """
    Instantiate a ranking method.

    :param name:
        The method name.
    :param kwargs:
        Additional keyword based arguments passed to the method's constructor.

    :return:
        An instatiated ranking method.
    """
    cls = get_subclass_by_name(base_class=RankingMethod, name=name, normalizer=name_normalizer)
    real_kwargs = dict()
    for key, value in kwargs.items():
        signature = inspect.signature(cls.__init__)
        if key in signature.parameters:
            real_kwargs[key] = value
        else:
            print(f"Unused argument {key}={value}")
    return cls(**real_kwargs)
