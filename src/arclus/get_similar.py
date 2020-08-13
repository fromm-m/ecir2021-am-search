from typing import Tuple

import torch


class Sim:
    def sim(self, claims: torch.Tensor, premises: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class KNNSim(Sim):
    def sim(self, claims: torch.Tensor, premises: torch.Tensor) -> torch.Tensor:
        # change distance to similarity
        return 1 / (1 + torch.cdist(claims, premises))


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
