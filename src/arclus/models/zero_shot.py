"""Implementation os baseline ranking methods."""
import logging
import pathlib
from abc import ABC
from collections import defaultdict
from logging import Logger
from operator import itemgetter
from typing import Collection, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import pandas
import torch
from sklearn.cluster import KMeans

from .base import RankingMethod
from ..evaluation import mndcg_score
from ..settings import (
    CLAIMS_TEST_FEATURES, PREMISES_TEST_FEATURES,
)
from ..similarity import Similarity, normalize_similarity
from ..utils import resolve_num_clusters

logger: Logger = logging.getLogger(__name__)


class ZeroShotRanking(RankingMethod, ABC):
    """Abstract base class for zero-shot methods."""

    # Pre-computed representations
    claims: Mapping[int, torch.FloatTensor]
    premises: Mapping[str, torch.FloatTensor]

    similarities: Collection[Union[Similarity, str]]
    similarity: Optional[Similarity]

    def __init__(
        self,
        similarities: Collection[Similarity],
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
        self.similarities = similarities
        self.similarity = None

        # Load pre-computed representations
        self.claims = torch.load(claims_path)
        self.premises = torch.load(premises_path)

    def _get_representations(self, claim_id: int, premise_ids: Sequence[str]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # get the claim representation
        claim_repr = self.claims[claim_id].unsqueeze(dim=0)
        # get premise representations
        premise_repr = torch.stack([self.premises[premise_id] for premise_id in premise_ids], dim=0)
        return claim_repr, premise_repr


def _rank_by_similarity(
    claim_repr: torch.FloatTensor,
    premise_repr: torch.FloatTensor,
    similarity: Similarity,
    k: int,
    premise_ids: Sequence[str],
) -> Sequence[str]:
    # corner case:
    k = min(k, premise_repr.shape[0])
    if k <= 0:
        return []
    # find most similar
    top_ids = similarity.sim(
        left=claim_repr,
        right=premise_repr,
    ).topk(k=k, largest=True, sorted=True).indices.squeeze(dim=0)
    # re-translate to original IDs
    return [premise_ids[i] for i in top_ids.tolist()]


class ZeroShotKNN(ZeroShotRanking):
    """Rank according to similarity of pre-trained BERT representations."""

    def fit(
        self,
        training_data: pandas.DataFrame,
        k: int,
    ) -> "RankingMethod":
        scores = defaultdict(list)
        for query_claim_id, group in training_data.groupby(by="claim_id"):
            premise_ids = group["premise_id"].tolist()
            claim_repr, premise_repr = self._get_representations(claim_id=query_claim_id, premise_ids=premise_ids)
            for similarity in self.similarities:
                y_pred = _rank_by_similarity(
                    claim_repr=claim_repr,
                    premise_repr=premise_repr,
                    similarity=normalize_similarity(similarity),
                    k=k,
                    premise_ids=premise_ids,
                )
                scores[similarity].append(mndcg_score(
                    y_pred=y_pred,
                    data=group,
                    k=k,
                ))
        # average over claims
        scores = _average_scores(scores=scores, num=len(training_data["claim_id"].unique()))
        self.similarity = normalize_similarity(similarity=max(scores.items(), key=itemgetter(1))[0])
        return self

    def rank(self, claim_id: int, premise_ids: Sequence[str], k: int) -> Sequence[str]:  # noqa: D102
        if self.similarity is None:
            raise ValueError(f"{self.__class__.__name__} must be fit before rank is called.")
        claim_repr, premise_repr = self._get_representations(claim_id=claim_id, premise_ids=premise_ids)
        return _rank_by_similarity(
            claim_repr=claim_repr,
            premise_repr=premise_repr,
            similarity=self.similarity,
            k=k,
            premise_ids=premise_ids,
        )


def _cluster_representations(
    premise_repr: torch.FloatTensor,
    ratio: float,
    k: int,
) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    # cluster premises
    algorithm = KMeans(n_clusters=resolve_num_clusters(ratio=ratio, num_premises=premise_repr.shape[0], k=k))
    cluster_assignment = torch.as_tensor(algorithm.fit_predict(premise_repr.numpy()))
    cluster_centers = torch.as_tensor(algorithm.cluster_centers_)
    return cluster_assignment, cluster_centers


def _choose_premises_from_clusters(
    claim_repr: torch.FloatTensor,
    premise_repr: torch.FloatTensor,
    premise_ids: Sequence[str],
    cluster_assignment: torch.LongTensor,
    cluster_centers: torch.FloatTensor,
    k: int,
    cluster_representative: str,
    similarity: Similarity,
) -> Sequence[str]:
    n_clusters = cluster_centers.shape[0]
    local_premise_ids = torch.arange(premise_repr.shape[0])
    repr_ids = torch.full(size=(n_clusters,), fill_value=-1, dtype=torch.long)
    for i in range(n_clusters):
        if cluster_representative == 'closest-to-center':
            anchor = cluster_centers[i].unsqueeze(dim=0)
        elif cluster_representative == 'closest-to-claim':
            anchor = claim_repr
        else:
            raise NotImplementedError(cluster_representative)
        mask = cluster_assignment == i
        if not mask.any():
            continue
        premises_in_cluster = premise_repr[mask]
        idx = similarity.sim(
            left=anchor,
            right=premises_in_cluster,
        ).argmax(dim=1)[0]
        repr_ids[i] = local_premise_ids[mask][idx]
    cluster_repr_id = repr_ids
    # find most similar clusters
    non_empty_clusters = cluster_repr_id >= 0
    # corner case when there are not enough clusters
    k = min(k, non_empty_clusters.sum())
    top_cluster_id = similarity.sim(
        left=claim_repr,
        right=premise_repr[cluster_repr_id[non_empty_clusters]],
    ).topk(k=k, largest=True, sorted=True).indices.squeeze(dim=0)
    # re-translate to local (batch) premise ID
    ids = [cluster_repr_id[non_empty_clusters][i] for i in top_cluster_id.tolist()]
    top_ids = ids
    return [premise_ids[i] for i in top_ids]


K = TypeVar("K")


def _average_scores(scores: Mapping[K, Collection[float]], num: int) -> Mapping[K, float]:
    return {
        key: sum(values) / num
        for key, values in scores.items()
    }


class ZeroShotClusterKNN(ZeroShotRanking):
    """Rank according to similarity of pre-trained BERT representations, return at most one premise for each cluster."""

    def __init__(
        self,
        similarities: Collection[Similarity],
        cluster_ratios: Sequence[float],
        claims_path: pathlib.Path = CLAIMS_TEST_FEATURES,
        premises_path: pathlib.Path = PREMISES_TEST_FEATURES,
        cluster_representatives: Collection[str] = ('closest-to-center', "closest-to-claim"),
    ):
        """
        Initialize the method.

        :param similarities:
            The similarity to use for the representations.
        :param cluster_ratios:
            The cluster ratio grid to evaluate.
        :param claims_path:
            The path to the pre-computed claims representations.
        :param premises_path:
            The path to the pre-computed premises representations.
        :param cluster_representatives:
            The method to choose a cluster representative. From {'closest-to-center', 'closest-to-claim'}.
        """
        super().__init__(similarities=similarities, claims_path=claims_path, premises_path=premises_path)
        self.cluster_representatives = cluster_representatives
        self.cluster_representative = None
        self.ratios = cluster_ratios
        self.ratio = None

    def fit(
        self,
        training_data: pandas.DataFrame,
        k: int,
    ) -> "RankingMethod":
        scores = defaultdict(list)
        for query_claim_id, group in training_data.groupby(by="claim_id"):
            premise_ids = group["premise_id"].tolist()
            claim_repr, premise_repr = self._get_representations(claim_id=query_claim_id, premise_ids=premise_ids)
            for ratio in self.ratios:
                cluster_assignment, cluster_centers = _cluster_representations(premise_repr, ratio=self.ratio, k=k)
                for similarity in self.similarities:
                    for representative in self.cluster_representatives:
                        y_pred = _choose_premises_from_clusters(
                            claim_repr=claim_repr,
                            premise_repr=premise_repr,
                            premise_ids=premise_ids,
                            cluster_assignment=cluster_assignment,
                            cluster_centers=cluster_centers,
                            k=k,
                            cluster_representative=representative,
                            similarity=normalize_similarity(similarity=similarity),
                        )
                        scores[ratio, similarity, representative].append(mndcg_score(y_pred=y_pred, data=group, k=k))
        # average over claims
        scores = _average_scores(scores=scores, num=len(training_data["claim_id"].unique()))
        self.ratio, self.similarity, self.cluster_representative = max(scores.items(), key=itemgetter(1))[0]
        self.similarity = normalize_similarity(similarity=self.similarity)
        return self

    def rank(self, claim_id: int, premise_ids: Sequence[str], k: int) -> Sequence[str]:  # noqa: D102
        if self.ratio is None or self.cluster_representative is None or self.similarity is None:
            raise ValueError(f"{self.__class__.__name__} must be fit before rank is called.")
        claim_repr, premise_repr = self._get_representations(claim_id, premise_ids)
        cluster_assignment, cluster_centers = _cluster_representations(premise_repr=premise_repr, ratio=self.ratio, k=k)
        return _choose_premises_from_clusters(
            claim_repr=claim_repr,
            premise_repr=premise_repr,
            premise_ids=premise_ids,
            cluster_assignment=cluster_assignment,
            cluster_centers=cluster_centers,
            k=k,
            cluster_representative=self.cluster_representative,
            similarity=self.similarity,
        )
