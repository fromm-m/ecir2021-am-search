"""Implementation os baseline ranking methods."""
import logging
import pathlib
from abc import ABC
from collections import defaultdict
from logging import Logger
from operator import itemgetter
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy
import pandas
import torch
from sklearn.cluster import KMeans

from .base import RankingMethod
from ..evaluation import mndcg_score
from ..settings import (
    CLAIMS_TEST_FEATURES, PREMISES_TEST_FEATURES, PREP_ASSIGNMENTS_TEST,
    PREP_TEST_PRODUCT_SIMILARITIES, PREP_TEST_SIMILARITIES, PREP_TEST_STATES,
)
from ..similarity import CosineSimilarity, Similarity, get_similarity_by_name
from ..utils_am import inference_no_args, load_bert_model_and_data_no_args

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
                left=anchor,
                right=premises_in_cluster,
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
            left=claim_repr,
            right=premise_repr[cluster_repr_id[non_empty_clusters]],
        ).topk(k=k, largest=True, sorted=True).indices.squeeze(dim=0)

        # re-translate to local (batch) premise ID
        top_ids = [cluster_repr_id[non_empty_clusters][i] for i in top_cluster_id.tolist()]

        # re-translate to original IDs
        return [premise_ids[i] for i in top_ids]


def _prepare_claim_similarities(
    cache_root: str,
    model_path: str,
    product: bool,
) -> pathlib.Path:
    """
    Prepare similarities.

    :param cache_root:
        The cache_root for the model.
    :param model_path:
        The path to the model.
    :param product:
        Whether to compute similarities to all result_claims, or only to the relevant query claims.

    :return:
        The path of the precomputed similarities.
    """
    output_path = PREP_TEST_PRODUCT_SIMILARITIES if product else PREP_TEST_SIMILARITIES
    if not Path(output_path).is_file():
        logger.info('computing similarities')
        # load bert model and the data
        batch_size = 180
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
            product=product,
        )

        # generate logits for all claims-premise pairs
        logger.info('Run inference')
        predictions = inference_no_args(
            data=data,
            loader=loader,
            logger=logger,
            model=model,
            batch_size=batch_size,
        )
        precomputed_similarities = dict(zip(guids, predictions))
        torch.save(precomputed_similarities, output_path)
    return output_path


def get_query_claim_similarities(
    sim: Mapping[Tuple[str, int], float],
    softmax: bool,
) -> Mapping[Tuple[str, int], float]:
    """
    Preprocess query claim similarities.

    :param sim:
        A mapping from (premise_id, claim_id) to the logits of the similarity model, shape: (2,).
    :param softmax:
        Whether to apply softmax or use raw logits.

    :return:
        A mapping from (premise_id, claim_id) to scalar similarity value.
    """
    # ensure consistent order
    pairs = sorted(sim.keys())

    # create tensor,shape: (num_pairs, 2)
    sim = torch.stack(
        tensors=[
            torch.as_tensor(data=sim[pair], dtype=torch.float32)
            for pair in pairs
        ],
        dim=0,
    )

    # apply softmax is requested
    if softmax:
        sim = sim.softmax(dim=-1)

    # take probability of "similar" class
    sim = sim[:, 1]

    # one row corresponds to one pair similarity
    return dict(zip(pairs, sim))


def get_premise_representations(
    sim: Mapping[Tuple[float, int], float],
    softmax: bool,
) -> Mapping[str, torch.FloatTensor]:
    """
    Construct premise representations as similarity vectors to claims.

    :param sim:
        The similarities for (premise_id, claim_id) pairs.
    :param softmax
        Whether to apply softmax or use raw logits.

    :return:
        A mapping from premise_id to a vector of similarities, shape: (num_claims,).
    """
    # verify that similarities are available for all claim, premise pairs
    premise_ids, claim_ids = [
        sorted(set(map(itemgetter(pos), sim.keys())))
        for pos in (0, 1)
    ]
    assert set(sim.keys()) == set(
        (pid, cid)
        for pid in premise_ids
        for cid in claim_ids
    )

    # convert to tensor, shape: (num_premises, num_claims, 2)
    sim = torch.stack(
        tensors=[
            torch.stack(
                tensors=[
                    torch.as_tensor(data=sim[premise_id, claim_id], dtype=torch.float32)
                    for claim_id in claim_ids
                ],
                dim=0,
            )
            for premise_id in premise_ids
        ],
        dim=0,
    )
    assert sim.shape == (len(premise_ids), len(claim_ids), 2)

    # apply softmax is requested
    if softmax:
        sim = sim.softmax(dim=-1)

    # take probability of "similar" class
    sim = sim[:, :, 1]

    # one row corresponds to one premise representation
    return dict(zip(premise_ids, sim))


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
        buffer_path_sim = similarities_dir / PREP_TEST_SIMILARITIES
        buffer_path_states = similarities_dir / PREP_TEST_STATES
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
            self.precomputed_similarities = {k: torch.softmax(torch.as_tensor(v, dtype=float), dim=-1)[1] for k, v in
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


def core_set(
    similarity: torch.FloatTensor,
    first_id: int,
    k: int,
    premise_bias: Optional[torch.FloatTensor] = None,
) -> Sequence[int]:
    """
    Coreset method based on full pairwise similarity matrix.

    :param similarity: shape: (n, n)
        The full pairwise similarity/score matrix. Larger values indicate better fit.
    :param first_id: >=0, <n
        The first chosen ID.
    :param k: >0
        The number of candidates to choose.
    :param premise_bias: shape: (n,)
        A bias for premise.

    :return:
        An ordered list of k selected candidate IDs.
    """
    n = similarity.shape[0]
    assert similarity.shape == (n, n)
    k = min(k, n)

    result = [first_id]

    chosen_mask = torch.zeros(n, dtype=torch.bool, device=similarity.device)
    chosen_mask[first_id] = True

    for i in range(1, k):
        # select similarity from candidates to chosen, shape: (num_cand, num_chosen)
        score = similarity[chosen_mask].t()[~chosen_mask]
        # largest similarity to chosen, shape: (num_cand), smallest similarity
        candidate_score = score.max(dim=-1).values
        # apply bias
        if premise_bias is not None:
            candidate_score = candidate_score - premise_bias[~chosen_mask]
        local_next_id = candidate_score.argmin()
        # convert to global id
        next_id = torch.arange(n, device=similarity.device)[~chosen_mask][local_next_id].item()

        # update mask
        chosen_mask[next_id] = True

        # append to result
        result.append(next_id)

    return result


class Coreset(LearnedSimilarityKNN):

    def __init__(
        self,
        model_path: str,
        similarities_dir: str,
        premise_premise_similarity: Similarity = CosineSimilarity(),
        cache_root: str = '/nfs/data3/obermeier/arclus/temp/',
        debug: bool = False,
        fill_to_k: bool = False,
    ):
        """
        Initialize the method.

        :param model_path:
            Directory where the fine-tuned bert similarity model checkpoint is located.
        :param cache_root:
            The directory where temporary BERT inference files are stored.
        """
        super().__init__(model_path=model_path, cache_root=cache_root, softmax=True, similarities_dir=similarities_dir)
        self.threshold = None
        if isinstance(premise_premise_similarity, str):
            premise_premise_similarity = get_similarity_by_name(premise_premise_similarity)
        self.premise_premise_similarity = premise_premise_similarity
        self.debug = debug
        self.fill_to_k = fill_to_k

    def fit(
        self,
        training_data: pandas.DataFrame,
        k: int,
    ):
        # compute scores for all thresholds
        scores = defaultdict(dict)
        thresholds = []
        claim_ids = training_data["claim_id"].unique().tolist()
        for claim_id, group in training_data.groupby(by="claim_id"):
            premise_ids = group["premise_id"].tolist()
            for threshold in sorted(set(
                self.precomputed_similarities[premise_id, claim_id]
                for premise_id in premise_ids
            )):
                # Do not fill for tuning threshold
                y_pred = self._rank(claim_id=claim_id, premise_ids=premise_ids, k=k, threshold=threshold, fill_to_k=False)
                scores[claim_id][threshold] = mndcg_score(y_pred=y_pred, data=group, k=k)
                thresholds.append(threshold)

        def _eval_threshold(threshold: float) -> float:
            score = 0.0
            for claim_id in claim_ids:
                # score(MAX_VALUE) = 0
                # score(MIN_VALUE) = score(min(threshold))
                other_threshold = min(
                    (
                        other_threshold
                        for other_threshold in scores[claim_id].keys()
                        if other_threshold > threshold
                    ),
                    default=None,
                )
                if other_threshold is not None:
                    score += scores[claim_id][other_threshold]
                # else:
                #     score += 0
            return score / len(claim_ids)

        # choose threshold
        if self.debug:
            thresholds = numpy.asarray(thresholds)
            _eval_threshold = numpy.vectorize(_eval_threshold)
            scores = _eval_threshold(thresholds)
            fold_hash = abs(hash(tuple(sorted(claim_ids))))
            numpy.save(f"/tmp/scores_k{k}_{self.premise_premise_similarity}_{fold_hash}.npy", numpy.stack([thresholds, scores]))
            self.threshold = thresholds[scores.argmax()]
        else:
            self.threshold = max(thresholds, key=_eval_threshold)

    def rank(self, claim_id: int, premise_ids: Sequence[str], k: int) -> Sequence[str]:  # noqa: D102
        if self.threshold is None:
            raise ValueError(f"{self.__class__.__name__} must be fit before rank is called.")

        return self._rank(claim_id=claim_id, premise_ids=premise_ids, k=k, threshold=self.threshold, fill_to_k=self.fill_to_k)

    def _rank(self, claim_id, premise_ids, k, threshold: float, fill_to_k: bool) -> Sequence[str]:
        def lookup_similarity(premise_id: str) -> float:
            """Get similarity between a premise and query claim."""
            return self.precomputed_similarities[premise_id, claim_id]

        # filter premise IDs
        premise_ids = [
            premise_id
            for premise_id in premise_ids
            if lookup_similarity(premise_id=premise_id) > threshold
        ]

        if len(premise_ids) < 1:
            logger.warning("No premise after thresholding.")

        chosen = []

        if len(premise_ids) > 0:
            # select first premise as closest to claim
            first_id = premise_ids.index(max(premise_ids, key=lookup_similarity))
            # get premise representations
            premise_repr = torch.stack(
                [
                    self.precomputed_states[p_id, claim_id]
                    for p_id in premise_ids
                ],
                dim=0,
            )

            # compute pair-wise similarity matrix
            similarity = self.premise_premise_similarity.sim(premise_repr, premise_repr)

            # apply coreset
            local_ids = core_set(similarity=similarity, first_id=first_id, k=k)

            # convert back to premise_ids
            chosen = [premise_ids[i] for i in local_ids]

        if fill_to_k and len(chosen) < k:
            chosen += sorted(
                set(premise_ids).difference(chosen),
                key=lookup_similarity,
                reverse=True,
            )[:k - len(chosen)]

        return chosen


class BiasedCoreset(LearnedSimilarityKNN):
    """Convex combination of coreset score and similarity to query."""

    def __init__(
        self,
        model_path: str,
        similarities_dir: str,
        premise_premise_similarity: Similarity = CosineSimilarity(),
        cache_root: str = '/nfs/data3/obermeier/arclus/temp/',
        debug: bool = False,
        resolution: int = 10,
    ):
        """
        Initialize the method.

        :param model_path:
            Directory where the fine-tuned bert similarity model checkpoint is located.
        :param cache_root:
            The directory where temporary BERT inference files are stored.
        """
        super().__init__(model_path=model_path, cache_root=cache_root, softmax=True, similarities_dir=similarities_dir)
        self.alpha = None
        if isinstance(premise_premise_similarity, str):
            premise_premise_similarity = get_similarity_by_name(premise_premise_similarity)
        self.premise_premise_similarity = premise_premise_similarity
        self.debug = debug
        self.resolution = resolution

    def fit(
        self,
        training_data: pandas.DataFrame,
        k: int,
    ):
        def _evaluate_alpha(alpha: float):
            score = 0.0
            for claim_id, group in training_data.groupby(by="claim_id"):
                premise_ids = group["premise_id"].tolist()
                y_pred = self._rank(claim_id=claim_id, premise_ids=premise_ids, k=k, alpha=alpha)
                score += mndcg_score(y_pred=y_pred, data=group, k=k)
            return score / len(training_data["claim_id"].unique())

        # compute scores for all alpha values
        alphas = numpy.linspace(start=0, stop=1, num=self.resolution)
        if self.debug:
            _eval_alpha = numpy.vectorize(_evaluate_alpha)
            scores = _eval_alpha(alphas)
            fold_hash = abs(hash(tuple(sorted(training_data["claim_id"].unique()))))
            numpy.save(f"/tmp/convex_scores_k{k}_{self.premise_premise_similarity}_{fold_hash}.npy", numpy.stack([alphas, scores]))
            self.alpha = alphas[scores.argmax()]
        else:
            self.alpha = max(alphas, key=_evaluate_alpha)

    def rank(self, claim_id: int, premise_ids: Sequence[str], k: int) -> Sequence[str]:  # noqa: D102
        if self.alpha is None:
            raise ValueError(f"{self.__class__.__name__} must be fit before rank is called.")

        return self._rank(claim_id=claim_id, premise_ids=premise_ids, k=k, alpha=self.alpha)

    def _rank(self, claim_id: int, premise_ids: Sequence[str], k: int, alpha: float) -> Sequence[str]:
        premise_ids = list(premise_ids)

        def lookup_similarity(premise_id: str) -> float:
            """Get similarity between a premise and query claim."""
            return self.precomputed_similarities[premise_id, claim_id]

        # select first premise as closest to claim
        first_id = premise_ids.index(max(premise_ids, key=lookup_similarity))

        # get premise representations
        premise_repr = torch.stack(
            [
                self.precomputed_states[p_id, claim_id]
                for p_id in premise_ids
            ],
            dim=0,
        )

        claim_premise_similarity = torch.as_tensor(
            data=[
                lookup_similarity(premise_id=premise_id)
                for premise_id in premise_ids
            ],
            dtype=torch.float32,
        )

        # compute pair-wise similarity matrix
        premise_premise_similarity = self.premise_premise_similarity.sim(premise_repr, premise_repr)

        # apply coreset
        local_ids = core_set(
            similarity=alpha * premise_premise_similarity,
            first_id=first_id,
            k=k,
            premise_bias=(1 - alpha) * claim_premise_similarity,
        )

        # convert back to premise_ids
        return [premise_ids[i] for i in local_ids]


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
        buffer_path = similarities_dir / PREP_TEST_SIMILARITIES
        buffer_path_product = similarities_dir / PREP_TEST_PRODUCT_SIMILARITIES
        logger.info(f'Using softmax: {softmax}')
        if not buffer_path_product.is_file():
            logger.info('computing similarities')
            # load bert model and the data
            logger.info('Load data')
            batch_size = 120
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
            # self.premise_representations = get_premise_representations(
            #    sim=torch.load(_prepare_claim_similarities(
            #        cache_root=cache_root,
            #        model_path=model_path,
            #       product=True)))

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
            self.precomputed_similarities = {k: torch.softmax(torch.FloatTensor(v, dtype=float), dim=-1) for k, v in
                                             self.precomputed_similarities.items()}
            self.precomputed_similarities_resultclaims = {k: torch.softmax(torch.FloatTensor(v, dtype=float), dim=-1)
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
        premise_repr = torch.stack([torch.FloatTensor([v for k, v in self.precomputed_similarities_resultclaims.items()
                                                       if k[0] == premise_id]) for premise_id in premise_ids], dim=0)

        return _premise_cluster_filtered(
            claim_id=claim_id,
            premise_ids=premise_ids,
            premise_repr=premise_repr,
            similarities=self.precomputed_similarities,
            k=k,
            ratio=self.ratio,
        )
