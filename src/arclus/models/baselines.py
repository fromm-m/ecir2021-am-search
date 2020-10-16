import logging
import pathlib
from abc import ABC
from logging import Logger
from operator import itemgetter
from typing import Mapping, Optional, Sequence, Tuple

import torch
from sklearn.cluster import KMeans
from arclus.models.base import RankingMethod
from arclus.settings import CLAIMS_TEST_FEATURES, PREMISES_TEST_FEATURES, PREP_ASSIGNMENTS_TEST, \
    PREP_TEST_PRODUCT_SIMILARITIES, PREP_TEST_SIMILARITIES, PREP_TEST_STATES
from arclus.similarity import Similarity
from arclus.utils_am import inference_no_args, load_bert_model_and_data_no_args

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
            claims=claim_repr,
            premises=premise_repr,
        ).topk(k=k, largest=True, sorted=True).indices.squeeze(dim=0)
        # re-translate to original IDs
        return [premise_ids[i] for i in top_ids.tolist()]


def _num_clusters(ratio: Optional[float], num_premises: int, k: int) -> int:
    if ratio is None:
        return k
    n_clusters = int(round(ratio * num_premises))
    n_clusters = max(n_clusters, k)
    n_clusters = min(n_clusters, num_premises)
    return n_clusters


class ZeroShotClusterKNN(ZeroShotRanking):
    """Rank according to similarity of pre-trained BERT representations, return at most one premise for each cluster."""

    def __init__(
        self,
        similarity: Similarity,
        cluster_ratio: Optional[float] = 0.5,
        claims_path: pathlib.Path = CLAIMS_TEST_FEATURES,
        premises_path: pathlib.Path = PREMISES_TEST_FEATURES,
        cluster_representative: str = 'closest-to-center',
    ):
        """
        Initialize the method.

        :param similarity:
            The similarity to use for the representations.
        :param cluster_ratio: >0
            The relative number of clusters to use. If None, use k.
        :param claims_path:
            The path to the pre-computed claims representations.
        :param premises_path:
            The path to the pre-computed premises representations.
        :param cluster_representative:
            The method to choose a cluster representative. From {'closest-to-center', 'closest-to-claim'}.
        """
        super().__init__(similarity=similarity, claims_path=claims_path, premises_path=premises_path)
        self.ratio = cluster_ratio
        self.cluster_representative = cluster_representative

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
                claims=anchor,
                premises=premises_in_cluster,
            ).argmax(dim=1)[0]
            repr_ids[i] = local_premise_ids[mask][idx]
        return repr_ids

    def rank(self, claim_id: int, premise_ids: Sequence[str], k: int) -> Sequence[str]:  # noqa: D102
        # get the claim representation
        claim_repr = self.claims[claim_id].unsqueeze(dim=0)

        # get premise representations
        premise_repr = torch.stack([self.premises[premise_id] for premise_id in premise_ids], dim=0)
        # cluster premises
        algorithm = KMeans(n_clusters=_num_clusters(ratio=self.ratio, num_premises=len(premise_ids), k=k))
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
            claims=claim_repr,
            premises=premise_repr[cluster_repr_id[non_empty_clusters]],
        ).topk(k=k, largest=True, sorted=True).indices.squeeze(dim=0)

        # re-translate to local (batch) premise ID
        top_ids = [cluster_repr_id[non_empty_clusters][i] for i in top_cluster_id.tolist()]

        # re-translate to original IDs
        return [premise_ids[i] for i in top_ids]


class LearnedSimilarityKNN(RankingMethod):
    """Rank premises according to precomputed fine-tuned BERT similarities for concatenation of premise and claim."""

    #: The precomputed similarities.
    precomputed_similarities: Mapping[Tuple[str, int], float]

    def __init__(
        self,
        softmax: bool,
        model_path: str,
        similarities_dir: str,
        cache_root: str = '/tmp/arclus/bert',
    ):
        """
        Initialize the method.

        :param softmax:
            Whether to apply softmax on the scores for the pairwise similarity model.
        :param model_path:
            Directory where the fine-tuned bert similarity model checkpoint is located.
        :param cache_root:
            The directory where temporary BERT inference files are stored.
        """
        buffer_path_sim = similarities_dir + PREP_TEST_SIMILARITIES
        buffer_path_states = similarities_dir + PREP_TEST_STATES
        logger.info(f'Using softmax: {softmax}')
        if not buffer_path_sim.is_file() or not buffer_path_states.is_file():
            logger.info('computing similarities')
            # load bert model and the data
            batch_size = 120
            logger.info('Load data')
            loader, data, model, guids = load_bert_model_and_data_no_args(
                model_path=model_path,
                task_name="SIM",
                batch_size=batch_size,
                data_dir=PREP_ASSIGNMENTS_TEST,
                overwrite_cache=True,
                max_seq_length=512,
                model_type="bert",
                cache_root=cache_root,
                product=False
            )

            # generate logits for all claims-premise pairs
            # predictions = inference(args, data, loader, logger, model)
            logger.info('Run inference')
            predictions, states = inference_no_args(
                data=data,
                loader=loader,
                logger=logger,
                model=model,
                batch_size=batch_size,
            )
            precomputed_similarities = dict(zip(guids, predictions))
            precomputed_states = dict(zip(guids, states))

            torch.save(precomputed_similarities, buffer_path_sim)
            torch.save(precomputed_states, buffer_path_states)

        self.precomputed_similarities = torch.load(buffer_path_sim)
        self.precomputed_states = torch.load(buffer_path_states)
        if softmax:
            self.precomputed_similarities = {k: torch.softmax(torch.tensor(v, dtype=float), dim=-1)[1] for k, v in
                                             self.precomputed_similarities.items()}
        else:
            self.precomputed_similarities = {k: v[1] for k, v in self.precomputed_similarities.items()}

    def rank(self, claim_id: int, premise_ids: Sequence[str], k: int) -> Sequence[str]:  # noqa: D102
        def lookup_similarity(premise_id: str) -> float:
            return self.precomputed_similarities[premise_id, claim_id]

        return sorted(premise_ids, key=lookup_similarity, reverse=True)[:k]


def _premise_cluster_filtered(
    claim_id: int,
    premise_ids: Sequence[str],
    premise_repr: torch.FloatTensor,
    similarities: Mapping[Tuple[str, int], float],
    k: int,
    ratio: Optional[float],
) -> Sequence[str]:
    """
    Return premises sorted by similarity to claim, filtered to contain at most one element per cluster.

    :param claim_id:
        The claim ID.
    :param premise_ids:
        The premise IDs.
    :param premise_repr: shape: (num_premises, dim)
        The corresponding representations.
    :param similarities:
        The pre-computed similarities.
    :param k:
        The number of premise IDs to return.
    :param ratio:
        The cluster ratio.

    :return:
        A list of (at most) k premise IDs.
    """
    num_premises = len(premise_ids)
    # cluster premises
    algorithm = KMeans(n_clusters=_num_clusters(ratio=ratio, num_premises=num_premises, k=k))
    cluster_assignment = algorithm.fit_predict(premise_repr.numpy()).tolist()

    def lookup_similarity(premise_id: str) -> float:
        return similarities[premise_id, claim_id]

    seen_clusters = set()
    result = []
    mapping = dict(zip(premise_ids, cluster_assignment))
    for premise_id in sorted(premise_ids, key=lookup_similarity, reverse=True):
        cluster_id = mapping[premise_id]
        if cluster_id not in seen_clusters:
            result.append(premise_id)
        seen_clusters.add(cluster_id)
    return result[:k]


class LearnedSimilarityClusterKNN(LearnedSimilarityKNN):
    """Rank premises according to precomputed fine-tuned BERT similarities for concatenation of premise and claim, only returning one premise for each cluster."""

    def __init__(
        self,
        cluster_ratio: float,
        softmax: bool,
        model_path: str,
        similarities_dir: str,
        cache_root: str = '/tmp/arclus/bert',
    ):
        """
        Initialize the method.

        :param cluster_ratio: >0
            The relative number of clusters to use. If None, use k.
        :param premises_path:
            The path where the precomputed premise representations are stored.
        :param softmax:
            Whether to apply softmax on the scores for the pairwise similarity model.
        :param model_path:
            Directory where the fine-tuned bert similarity model checkpoint is located.
        :param cache_root:
            The directory where temporary BERT inference files are stored.
        """
        super().__init__(model_path=model_path, cache_root=cache_root, softmax=softmax,
                         similarities_dir=similarities_dir)
        self.ratio = cluster_ratio

    def rank(self, claim_id: int, premise_ids: Sequence[str], k: int) -> Sequence[str]:  # noqa: D102
        # get premise representations
        premise_repr = torch.stack([self.precomputed_states[premise_id, claim_id] for premise_id in premise_ids], dim=0)

        return _premise_cluster_filtered(
            claim_id=claim_id,
            premise_ids=premise_ids,
            premise_repr=premise_repr,
            similarities=self.precomputed_similarities,
            k=k,
            ratio=self.ratio,
        )


class LearnedSimilarityMatrixClusterKNN(RankingMethod):
    """Rank premises according to precomputed fine-tuned BERT similarities for concatenation of premise and claim, only returning one premise for each cluster."""

    """Rank premises according to precomputed fine-tuned BERT similarities for concatenation of premise and claim."""

    #: The precomputed similarities.
    precomputed_similarities: Mapping[Tuple[str, int], float]

    def __init__(
        self,
        cluster_ratio: float,
        softmax: bool,
        model_path: str,
        similarities_dir: str,
        cache_root: str = '/tmp/arclus/bert',
    ):
        """
        Initialize the method.

        :param softmax:
            Whether to apply softmax on the scores for the pairwise similarity model.
        :param model_path:
            Directory where the fine-tuned bert similarity model checkpoint is located.
        :param cache_root:
            The directory where temporary BERT inference files are stored.
        """
        buffer_path = similarities_dir + PREP_TEST_SIMILARITIES
        buffer_path_product = similarities_dir + PREP_TEST_PRODUCT_SIMILARITIES
        logger.info(f'Using softmax: {softmax}')
        if not buffer_path_product.is_file():
            logger.info('computing similarities')
            # load bert model and the data
            batch_size = 120
            logger.info('Load data')
            loader, data, model, guids = load_bert_model_and_data_no_args(
                model_path=model_path,
                task_name="SIM",
                batch_size=batch_size,
                data_dir=PREP_ASSIGNMENTS_TEST,
                overwrite_cache=True,
                max_seq_length=512,
                model_type="bert",
                cache_root=cache_root,
                product=True,
            )

            # generate logits for all claims-premise pairs
            logger.info('Run inference')
            predictions, _ = inference_no_args(
                data=data,
                loader=loader,
                logger=logger,
                model=model,
                batch_size=batch_size,
            )
            precomputed_similarities = dict(zip(guids, predictions))
            torch.save(precomputed_similarities, buffer_path_product)

        self.precomputed_similarities = torch.load(buffer_path)
        self.precomputed_similarities_resultclaims = torch.load(buffer_path_product)
        if softmax:
            self.precomputed_similarities = {k: torch.softmax(torch.tensor(v, dtype=float), dim=-1) for k, v in
                                             self.precomputed_similarities.items()}
            self.precomputed_similarities_resultclaims = {k: torch.softmax(torch.tensor(v, dtype=float), dim=-1)
                                                          for k, v in
                                                          self.precomputed_similarities_resultclaims.items()}

        self.precomputed_similarities = {k: v[1] for k, v in self.precomputed_similarities.items()}
        self.precomputed_similarities_resultclaims = {k: v[1] for k, v in
                                                      self.precomputed_similarities_resultclaims.items()}
        # verify that similarities are available for all claim, premise pairs
        premise_ids, claim_ids = [
            sorted(set(map(itemgetter(pos), self.precomputed_similarities_resultclaims.keys())))

            for pos in (0, 1)
        ]
        assert set(self.precomputed_similarities_resultclaims.keys()) == set(
            (pid, cid) for pid in premise_ids for cid in claim_ids)

        # prepare premise representations; make sure that claims are always in the same order
        self.premise_representations = {
            premise_id: torch.as_tensor(
                data=[
                    self.precomputed_similarities_resultclaims[premise_id, claim_id]
                    for claim_id in claim_ids
                ],
                dtype=torch.float32,
            )
            for premise_id in premise_ids
        }

        self.ratio = cluster_ratio

    def rank(self, claim_id: int, premise_ids: Sequence[str], k: int) -> Sequence[str]:  # noqa: D102
        # get premise representations, as similarity vector to all claims
        premise_repr = torch.stack([torch.tensor([v for k, v in self.precomputed_similarities_resultclaims.items()
                                                  if k[0] == premise_id]) for premise_id in premise_ids], dim=0)

        return _premise_cluster_filtered(
            claim_id=claim_id,
            premise_ids=premise_ids,
            premise_repr=premise_repr,
            similarities=self.precomputed_similarities,
            k=k,
            ratio=self.ratio,
        )
