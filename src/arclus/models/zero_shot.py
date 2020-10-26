"""Implementation os baseline ranking methods."""
import logging
import pathlib
from abc import ABC
from logging import Logger
from operator import itemgetter
from typing import Mapping, Sequence

import pandas
import torch
from sklearn.cluster import KMeans

from .base import RankingMethod
from ..evaluation import mndcg_score
from ..settings import (
    CLAIMS_TEST_FEATURES, PREMISES_TEST_FEATURES,
)
from ..similarity import Similarity
from ..utils import resolve_num_clusters

logger: Logger = logging.getLogger(__name__)


class ZeroShotRanking(RankingMethod, ABC):
    """Abstract base class for zero-shot methods."""

    # Pre-computed representations
    claims: Mapping[int, torch.FloatTensor]
    premises: Mapping[str, torch.FloatTensor]

    def __init__(
        self,
        similarity: Similarity,
        claims_path: pathlib.Path = CLAIMS_TEST_FEATURES,
        premises_path: pathlib.Path = PREMISES_TEST_FEATURES,
    ):
        """
        Initialize the method.

        :param similarity:
            The similarity to use for the representations.
        :param claims_path:
            The path to the pre-computed claims representations.
        :param premises_path:
            The path to the pre-computed premises representations.
        """
        self.similarity = similarity

        # Load pre-computed representations
        self.claims = torch.load(claims_path)
        self.premises = torch.load(premises_path)


class ZeroShotKNN(ZeroShotRanking):
    """Rank according to similarity of pre-trained BERT representations."""

    def rank(self, claim_id: int, premise_ids: Sequence[str], k: int) -> Sequence[str]:  # noqa: D102
        # get the claim representation
        claim_repr = self.claims[claim_id].unsqueeze(dim=0)
        # get premise representations
        premise_repr = torch.stack([self.premises[premise_id] for premise_id in premise_ids], dim=0)
        # find most similar
        top_ids = self.similarity.sim(
            left=claim_repr,
            right=premise_repr,
        ).topk(k=k, largest=True, sorted=True).indices.squeeze(dim=0)
        # re-translate to original IDs
        return [premise_ids[i] for i in top_ids.tolist()]


class ZeroShotClusterKNN(ZeroShotRanking):
    """Rank according to similarity of pre-trained BERT representations, return at most one premise for each cluster."""

    def __init__(
        self,
        similarity: Similarity,
        cluster_ratios: Sequence[float],
        claims_path: pathlib.Path = CLAIMS_TEST_FEATURES,
        premises_path: pathlib.Path = PREMISES_TEST_FEATURES,
        cluster_representative: str = 'closest-to-center',
    ):
        """
        Initialize the method.

        :param similarity:
            The similarity to use for the representations.
        :param cluster_ratios:
            The cluster ratio grid to evaluate.
        :param claims_path:
            The path to the pre-computed claims representations.
        :param premises_path:
            The path to the pre-computed premises representations.
        :param cluster_representative:
            The method to choose a cluster representative. From {'closest-to-center', 'closest-to-claim'}.
        """
        super().__init__(similarity=similarity, claims_path=claims_path, premises_path=premises_path)
        self.cluster_representative = cluster_representative
        self.ratios = cluster_ratios
        self.ratio = None

    def fit(
        self,
        training_data: pandas.DataFrame,
        k: int,
    ) -> "RankingMethod":
        score = {
            ratio: []
            for ratio in self.ratios
        }
        num_query_claims = len(training_data["claim_id"].unique())
        for query_claim_id, group in training_data.groupby(by="claim_id"):
            # get the claim representation
            claim_repr = self.claims[query_claim_id].unsqueeze(dim=0)
            # get premise representations
            premise_ids = group["premise_id"].tolist()
            premise_repr = torch.stack([self.premises[premise_id] for premise_id in premise_ids], dim=0)
            for ratio in self.ratios:
                score[ratio].append(
                    mndcg_score(
                        y_pred=self._rank(
                            claim_repr=claim_repr,
                            premise_ids=premise_ids,
                            premise_repr=premise_repr,
                            k=k,
                        ),
                        data=group,
                        k=k,
                    )
                )
        # average over claims
        score = {
            ratio: sum(scores) / num_query_claims
            for ratio, scores in score.items()
        }
        self.ratio = max(score.items(), key=itemgetter(1))[0]
        return self

    def _get_cluster_representatives(
        self,
        claim_repr: torch.FloatTensor,
        premise_repr: torch.FloatTensor,
        assignment: torch.LongTensor,
        centroids: torch.FloatTensor,
    ) -> torch.LongTensor:
        n_clusters = centroids.shape[0]
        local_premise_ids = torch.arange(premise_repr.shape[0])
        repr_ids = torch.full(size=(n_clusters,), fill_value=-1, dtype=torch.long)
        for i in range(n_clusters):
            if self.cluster_representative == 'closest-to-center':
                anchor = centroids[i].unsqueeze(dim=0)
            elif self.cluster_representative == 'closest-to-claim':
                anchor = claim_repr
            else:
                raise NotImplementedError(self.cluster_representative)
            mask = assignment == i
            if not mask.any():
                continue
            premises_in_cluster = premise_repr[mask]
            idx = self.similarity.sim(
                left=anchor,
                right=premises_in_cluster,
            ).argmax(dim=1)[0]
            repr_ids[i] = local_premise_ids[mask][idx]
        return repr_ids

    def rank(self, claim_id: int, premise_ids: Sequence[str], k: int) -> Sequence[str]:  # noqa: D102
        if self.ratio is None:
            raise ValueError(f"{self.__class__.__name__} must be fit before rank is called.")

        # get the claim representation
        claim_repr = self.claims[claim_id].unsqueeze(dim=0)

        # get premise representations
        premise_repr = torch.stack([self.premises[premise_id] for premise_id in premise_ids], dim=0)

        return self._rank(
            claim_repr=claim_repr,
            premise_ids=premise_ids,
            premise_repr=premise_repr,
            k=k,
        )

    def _rank(
        self,
        claim_repr: torch.FloatTensor,
        premise_ids: Sequence[str],
        premise_repr: torch.FloatTensor,
        k: int,
    ) -> Sequence[str]:
        # cluster premises
        algorithm = KMeans(n_clusters=resolve_num_clusters(ratio=self.ratio, num_premises=len(premise_ids), k=k))
        cluster_assignment = torch.as_tensor(algorithm.fit_predict(premise_repr.numpy()))
        cluster_centers = torch.as_tensor(algorithm.cluster_centers_)
        # choose representatives
        cluster_repr_id = self._get_cluster_representatives(
            claim_repr=claim_repr,
            premise_repr=premise_repr,
            assignment=cluster_assignment,
            centroids=cluster_centers,
        )
        # find most similar clusters
        non_empty_clusters = cluster_repr_id >= 0
        top_cluster_id = self.similarity.sim(
            left=claim_repr,
            right=premise_repr[cluster_repr_id[non_empty_clusters]],
        ).topk(k=k, largest=True, sorted=True).indices.squeeze(dim=0)
        # re-translate to local (batch) premise ID
        top_ids = [cluster_repr_id[non_empty_clusters][i] for i in top_cluster_id.tolist()]
        # re-translate to original IDs
        return [premise_ids[i] for i in top_ids]
