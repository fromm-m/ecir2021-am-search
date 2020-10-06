"""Basic interface for ranking methods."""
from abc import abstractmethod
from typing import Sequence


class RankingMethod:
    """Base class for ranking methods."""

    @abstractmethod
    def rank(
        self,
        claim_id: int,
        premise_ids: Sequence[str],
        k: int,
    ) -> Sequence[str]:
        """
        Return a sorted list of k premise IDs for each query claim.

        :param claim_id:
            The query claim ID.
        :param premise_ids:
            The candidate premise IDs.
        :param k: >0
            The number of premises to return for each claim.

        :return:
            The sorted list of k premise IDs for each query claim.
        """
        raise NotImplementedError
