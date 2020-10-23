"""Similarity functions for claims and premises."""
from typing import Tuple

import torch
from torch.nn import functional


class Similarity:
    """Base class for pairwise similarities."""

    def sim(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Compute similarity between all pairs of vectors."""
        raise NotImplementedError


class LpSimilarity(Similarity):
    """Similarity based on Lp distance."""

    def __init__(self, p: int = 2):
        """
        Initialize the similarity.

        :param p:
            The parameter p of the underlying Lp distance measure.
        """
        self.p = p

    def sim(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:  # noqa: D102
        # change distance to similarity
        return 1 / (1 + torch.cdist(left, right, p=self.p))

    def __str__(self) -> str:
        return f"l{self.p}"


class CosineSimilarity(Similarity):
    """Cosine similarity."""

    def sim(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return functional.normalize(left, p=2, dim=-1) @ functional.normalize(right, p=2, dim=-1).t()

    def __str__(self) -> str:
        return "cos"


def _mean_top_sim(sim: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    """Compute the mean similarity of the top-k matches along an axis."""
    return sim.topk(k=k, dim=dim, largest=True, sorted=False)[0].mean(dim=dim).unsqueeze(dim=dim)


class CSLSSimilarity(Similarity):
    """
    Apply CSLS normalization to similarity

    .. math ::
        csls[i, j] = 2 * sim[i, j] - avg(top_k(sim[i, :])) - avg(top_k(sim[:, j]))
    """

    def __init__(self, base: Similarity, k: int = 1):
        """
        Initialize the similarity.

        :param base:
            The base similarity.
        :param k:
            The parameter k controlling the "smoothing" effect.
        """
        self.base = base
        self.k = k

    def sim(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:  # noqa: D102
        # compute base similarity
        sim = self.base.sim(left=left, right=right)
        # normalize similarity
        return (2 * sim) - _mean_top_sim(sim=sim, k=self.k, dim=0) - _mean_top_sim(sim=sim, k=self.k, dim=1)


def get_most_similar(
    claims: torch.Tensor,
    premises: torch.Tensor,
    k: int,
    similarity: Similarity
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return most similar premises ranked in descending order.
    :param similarity: The similarity function
    :param claims: Representation of a claim
    :param premises: List of representations of premises
    :param k: How many premises to return
    :return: Most similar premises ranked in descending order and their indices
    """
    sim_values, indices = similarity.sim(left=claims, right=premises).topk(k=k, largest=True, sorted=True)

    return premises[indices], indices


_SIMILARITIES = dict(
    l2=LpSimilarity(p=2),
    l1=LpSimilarity(p=1),
    cos=CosineSimilarity(),
)


def get_similarity_by_name(
    name: str,
) -> Similarity:
    """Instantiate similarity."""
    return _SIMILARITIES[name.lower()]
