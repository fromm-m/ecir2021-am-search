from typing import Tuple

import torch
from torch.nn import functional


class Sim:
    def sim(self, claims: torch.Tensor, premises: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class LpSimilarity(Sim):
    def __init__(self, p: int = 2):
        self.p = p

    def sim(self, claims: torch.Tensor, premises: torch.Tensor) -> torch.Tensor:
        # change distance to similarity
        return 1 / (1 + torch.cdist(claims, premises, p=p))


class CosineSimilarity(Sim):
    def sim(self, claims: torch.Tensor, premises: torch.Tensor) -> torch.Tensor:
        return functional.normalize(claims, p=2, dim=-1) @ functional.normalize(premises, p=2, dim=-1).transpose()


def _mean_top_sim(sim: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    return sim.topk(k=k, dim=dim, largest=True, sorted=False)[0].mean(dim=dim).unsqueeze(dim=dim)


class CSLSSimilarity(Sim):
    """
    Apply CSLS normalization to similarity

    .. math ::
        csls[i, j] = 2 * sim[i, j] - avg(top_k(sim[i, :])) - avg(top_k(sim[:, j]))
    """

    def __init__(self, base: Sim, k: int = 1):
        self.base = base
        self.k = k

    def sim(self, claims: torch.Tensor, premises: torch.Tensor) -> torch.Tensor:
        sim = self.base.sim(claims=claims, premises=premises)
        return (2 * sim) - _mean_top_sim(sim=sim, k=self.k, dim=0) - _mean_top_sim(sim=sim, k=self.k, dim=1)


def get_most_similar(
    claims: torch.Tensor,
    premises: torch.Tensor,
    k: int,
    similarity: Sim
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return most similar premises ranked in descending order.
    :param similarity: The similarity function
    :param claims: Representation of a claim
    :param premises: List of representations of premises
    :param k: How many premises to return
    :return: Most similar premises ranked in descending order and their indices
    """
    sim_values, indices = similarity.sim(claims=claims, premises=premises).topk(k=k, largest=True, sorted=True)

    return premises[indices], indices
