"""Implementation os baseline ranking methods."""
import logging
import pathlib
from abc import ABC
from collections import defaultdict
from logging import Logger
from operator import itemgetter
from typing import Callable, Mapping, Optional, Sequence, Tuple

import numpy
import pandas
import torch
from sklearn.cluster import KMeans

from .base import RankingMethod
from ..evaluation import mndcg_score
from ..settings import (
    CLAIMS_TEST_FEATURES, PREMISES_TEST_FEATURES, PREP_ASSIGNMENTS_TEST, PREP_TEST_PRODUCT_SIMILARITIES, PREP_TEST_SIMILARITIES, PREP_TEST_STATES,
)
from ..similarity import CosineSimilarity, Similarity, get_similarity_by_name
from ..utils_am import inference_no_args, load_bert_model_and_data_no_args

logger: Logger = logging.getLogger(__name__)


def _num_clusters(ratio: Optional[float], num_premises: int, k: int) -> int:
    if ratio is None:
        return k
    n_clusters = int(round(ratio * num_premises))
    n_clusters = max(n_clusters, k)
    n_clusters = min(n_clusters, num_premises)
    return n_clusters


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


def _premise_cluster_filtered(
    premise_ids: Sequence[str],
    premise_repr: torch.FloatTensor,
    k: int,
    ratio: Optional[float],
    similarity_lookup: Callable[[str], float],
) -> Sequence[str]:
    """
    Return premises sorted by similarity to claim, filtered to contain at most one element per cluster.

    :param premise_ids:
        The premise IDs.
    :param premise_repr: shape: (num_premises, dim)
        The corresponding representations.
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

    seen_clusters = set()
    result = []
    mapping = dict(zip(premise_ids, cluster_assignment))
    for premise_id in sorted(premise_ids, key=similarity_lookup, reverse=True):
        cluster_id = mapping[premise_id]
        if cluster_id not in seen_clusters:
            result.append(premise_id)
        seen_clusters.add(cluster_id)
    return result[:k]


def _load_or_compute_similarities(
    cache_root: str,
    model_path: str,
    similarities_dir: pathlib.Path,
    softmax: bool = True,
    product: bool = True,
    with_states: bool = False,
) -> Tuple[Mapping[Tuple[str, int], float], Optional[Mapping[Tuple[str, int], torch.FloatTensor]]]:
    """
    Load the predicted similarities for all possible (premise_id, claim_id) pairs.

    :param cache_root:
        The cache root.
    :param model_path:
        The model path.
    :param similarities_dir:
        The similarities directory.

    :return:
        A mapping (premise_id, claim_id) -> similarity.
    """
    if product:
        buffer_path = similarities_dir / PREP_TEST_PRODUCT_SIMILARITIES
    else:
        buffer_path = similarities_dir / PREP_TEST_SIMILARITIES
    buffer_path_states = precomputed_states = None
    if with_states:
        if product:
            raise NotImplementedError
        buffer_path_states = similarities_dir / PREP_TEST_STATES

    if not buffer_path.is_file() or (with_states and not buffer_path_states.is_file()):
        logger.info("Computing similarities")

        # load bert model and the data
        logger.info("Load data")
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

        # generate logits for all claims-premise pairs
        logger.info("Run inference")
        predictions, states = inference_no_args(
            data=data,
            loader=loader,
            logger=logger,
            model=model,
            batch_size=batch_size,
        )

        # save artifact
        precomputed_similarities = dict(zip(guids, predictions))
        torch.save(precomputed_similarities, buffer_path)
        logger.info(f"Saved similarities to {buffer_path}")

        if with_states:
            precomputed_states = dict(zip(guids, states))
            torch.save(precomputed_states, buffer_path_states)
            logger.info(f"Saved states to {buffer_path_states}")
    else:
        logger.info(f"Loading precomputed similarities from {buffer_path}")
        precomputed_similarities = torch.load(buffer_path)

        if with_states:
            logger.info(f"Loading precomputed states from {buffer_path_states}")
            precomputed_states = torch.load(buffer_path_states)

    precomputed_similarities = get_query_claim_similarities(sim=precomputed_similarities, softmax=softmax)
    return precomputed_similarities, precomputed_states


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


class LearnedSimilarityBasedMethod(RankingMethod, ABC):
    """Base class for ranking methods based on learned similarity between claims and premises."""

    #: The precomputed similarities.
    precomputed_similarities: Mapping[Tuple[str, int], float]

    #: The precomputed representations
    precomputed_states: Optional[Mapping[Tuple[str, int], torch.FloatTensor]]

    def __init__(
        self,
        softmax: bool,
        model_path: str,
        similarities_dir: str,
        cache_root: Optional[str] = None,
        with_states: bool = True,
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
        if cache_root is None:
            cache_root = '/tmp/arclus/bert'
        self.precomputed_similarities, self.precomputed_states = _load_or_compute_similarities(
            cache_root=cache_root,
            model_path=model_path,
            similarities_dir=similarities_dir,
            softmax=softmax,
            product=False,
            with_states=with_states,
        )

    def _get_premise_representations(self, claim_id: int, premise_ids: Sequence[str]) -> torch.FloatTensor:
        """Get premise representations."""
        return torch.stack(
            [
                self.precomputed_states[premise_id, claim_id]
                for premise_id in premise_ids
            ],
            dim=0)

    def similarity_lookup(self, for_claim_id: int) -> Callable[[str], float]:
        """Create a similarity lookup for premises, with a fixed claim."""

        def lookup_similarity(premise_id: str) -> float:
            """Get similarity between a premise and query claim."""
            return self.precomputed_similarities[premise_id, for_claim_id]

        return lookup_similarity


class LearnedSimilarityKNN(LearnedSimilarityBasedMethod):
    """Rank premises according to precomputed fine-tuned BERT similarities for concatenation of premise and claim."""

    def rank(self, claim_id: int, premise_ids: Sequence[str], k: int) -> Sequence[str]:  # noqa: D102
        return sorted(premise_ids, key=self.similarity_lookup(for_claim_id=claim_id), reverse=True)[:k]


class LearnedSimilarityClusterKNN(LearnedSimilarityBasedMethod):
    """Rank premises according to precomputed fine-tuned BERT similarities for concatenation of premise and claim, only returning one premise for each cluster."""

    def __init__(
        self,
        cluster_ratio: float,
        softmax: bool,
        model_path: str,
        similarities_dir: str,
        cache_root: Optional[str] = None,
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
        super().__init__(
            softmax=softmax,
            model_path=model_path,
            similarities_dir=similarities_dir,
            cache_root=cache_root,
        )
        self.ratio = cluster_ratio

    def rank(self, claim_id: int, premise_ids: Sequence[str], k: int) -> Sequence[str]:  # noqa: D102
        return _premise_cluster_filtered(
            premise_ids=premise_ids,
            premise_repr=self._get_premise_representations(claim_id=claim_id, premise_ids=premise_ids),
            k=k,
            ratio=self.ratio,
            similarity_lookup=self.similarity_lookup(for_claim_id=claim_id),
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

        # largest similarity to chosen, shape: (num_cand)
        candidate_score = score.max(dim=-1).values

        # apply bias
        if premise_bias is not None:
            candidate_score = candidate_score - premise_bias[~chosen_mask]

        # choose highest score
        local_next_id = candidate_score.argmin()

        # convert to global id
        next_id = torch.arange(n, device=similarity.device)[~chosen_mask][local_next_id].item()

        # update mask
        chosen_mask[next_id] = True

        # append to result
        result.append(next_id)

    return result


class BaseCoreSetRanking(LearnedSimilarityBasedMethod):
    def __init__(
        self,
        model_path: str,
        similarities_dir: str,
        premise_premise_similarity: Similarity = CosineSimilarity(),
        cache_root: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize the method.

        :param model_path:
            Directory where the fine-tuned bert similarity model checkpoint is located.
        :param cache_root:
            The directory where temporary BERT inference files are stored.
        """
        super().__init__(model_path=model_path, cache_root=cache_root, softmax=True, similarities_dir=similarities_dir)
        if isinstance(premise_premise_similarity, str):
            premise_premise_similarity = get_similarity_by_name(premise_premise_similarity)
        self.premise_premise_similarity = premise_premise_similarity
        self.debug = debug

    def _get_pairwise_similarity_and_first_premise(
        self,
        claim_id,
        premise_ids,
    ) -> Tuple[int, torch.FloatTensor]:
        """
        Get pairwise premise similarities and select first premise by premise-claim similarity.

        :param claim_id:
            The claim ID.
        :param premise_ids:
            The premise IDs.

        :return:
            The first premise ID (as local ID, i.e. 0 <= local_premise_id < num_premises), and a
            (num_premises, num_premises) similarity matrix.
        """
        # select first premise as closest to claim
        first_id = premise_ids.index(max(premise_ids, key=self.similarity_lookup(for_claim_id=claim_id)))

        # get premise representations
        premise_repr = self._get_premise_representations(claim_id=claim_id, premise_ids=premise_ids)

        # compute pair-wise similarity matrix
        similarity = self.premise_premise_similarity.sim(left=premise_repr, right=premise_repr)

        return first_id, similarity


class Coreset(BaseCoreSetRanking):

    def __init__(
        self,
        model_path: str,
        similarities_dir: str,
        premise_premise_similarity: Similarity = CosineSimilarity(),
        cache_root: Optional[str] = None,
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
        super().__init__(
            model_path=model_path,
            similarities_dir=similarities_dir,
            premise_premise_similarity=premise_premise_similarity,
            cache_root=cache_root,
            debug=debug,
        )
        self.threshold = None
        self.fill_to_k = fill_to_k

    def fit(
        self,
        training_data: pandas.DataFrame,
        k: int,
    ):  # noqa: D102
        # we exploit that a change occurs only, if a threshold surpasses a measured similarity value
        # since the total score is the sum over the scores for individual query claims, we can independently compute
        # the score for each query claim at all points where the score would change, and "interpolate" the rest.
        scores = defaultdict(dict)
        thresholds = set()
        claim_ids = training_data["claim_id"].unique().tolist()
        for claim_id, group in training_data.groupby(by="claim_id"):
            premise_ids = group["premise_id"].tolist()

            # get all thresholds for which a change in the result _for this query claim_ occurs
            this_thresholds = set(
                self.precomputed_similarities[premise_id, claim_id]
                for premise_id in premise_ids
            )
            thresholds.update(this_thresholds)

            # Evaluate each threshold
            for threshold in this_thresholds:
                scores[claim_id][threshold] = mndcg_score(
                    y_pred=self._rank(
                        claim_id=claim_id,
                        premise_ids=premise_ids,
                        k=k,
                        threshold=threshold,
                        # Do not fill for tuning threshold
                        fill_to_k=False,
                    ),
                    data=group,
                    k=k
                )

        def _eval_threshold(threshold: float) -> float:
            # Since the filtering is given by
            #   [
            #       p
            #       for p in premises
            #       if sim(claim, p) > threshold
            #   ]
            # a value t between two adjacency threshold t_low and t_high behaves as the higher threshold
            # Thus, the score of t is equal to the smallest threshold t' which is larger than t
            # If t is chosen larger than the largest threshold, no premises remains. Thus, the score is 0.
            score = 0.0
            for claim_id in claim_ids:
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
                # else: score += 0
            # normalize score for interpretable results
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
        # filter premise IDs
        premise_ids = [
            premise_id
            for premise_id in premise_ids
            if self.precomputed_similarities[premise_id, claim_id] > threshold
        ]

        if len(premise_ids) > 0:
            first_id, similarity = self._get_pairwise_similarity_and_first_premise(
                claim_id=claim_id,
                premise_ids=premise_ids,
            )

            # apply coreset
            local_ids = core_set(similarity=similarity, first_id=first_id, k=k)

            # convert back to premise_ids
            chosen = [premise_ids[i] for i in local_ids]
        else:
            logger.warning("No premise after thresholding.")
            chosen = []

        if fill_to_k and len(chosen) < k:
            chosen += sorted(
                set(premise_ids).difference(chosen),
                key=self.similarity_lookup(for_claim_id=claim_id),
                reverse=True,
            )[:k - len(chosen)]

        return chosen


class BiasedCoreset(BaseCoreSetRanking):
    """Convex combination of coreset score and similarity to query."""

    def __init__(
        self,
        model_path: str,
        similarities_dir: str,
        premise_premise_similarity: Similarity = CosineSimilarity(),
        cache_root: Optional[str] = None,
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
        super().__init__(
            model_path=model_path,
            similarities_dir=similarities_dir,
            premise_premise_similarity=premise_premise_similarity,
            cache_root=cache_root,
            debug=debug,
        )
        self.alpha = None
        self.resolution = resolution

    def fit(
        self,
        training_data: pandas.DataFrame,
        k: int,
    ):
        def _evaluate_alpha(alpha: float) -> float:
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

        # select first premise as closest to claim and get pairwise premise similarities
        first_id, premise_premise_similarity = self._get_pairwise_similarity_and_first_premise(
            claim_id=claim_id,
            premise_ids=premise_ids,
        )

        claim_premise_similarity = torch.as_tensor(
            data=list(map(self.similarity_lookup(for_claim_id=claim_id), premise_ids)),
            dtype=torch.float32,
        )

        # apply coreset
        local_ids = core_set(
            similarity=alpha * premise_premise_similarity,
            first_id=first_id,
            k=k,
            premise_bias=(1 - alpha) * claim_premise_similarity,
        )

        # convert back to premise_ids
        return [premise_ids[i] for i in local_ids]


class LearnedSimilarityMatrixClusterKNN(LearnedSimilarityBasedMethod):
    """Rank premises according to precomputed fine-tuned BERT similarities for concatenation of premise and claim, only returning one premise for each cluster."""

    def __init__(
        self,
        cluster_ratio: float,
        softmax: bool,
        model_path: str,
        similarities_dir: str,
        cache_root: Optional[str] = None,
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
        super().__init__(
            softmax=softmax,
            model_path=model_path,
            similarities_dir=similarities_dir,
            cache_root=cache_root,
            with_states=False,
        )
        sim_premise_to_all_claims = _load_or_compute_similarities(
            cache_root=cache_root,
            model_path=model_path,
            similarities_dir=similarities_dir,
            softmax=softmax,
            product=True,
            with_states=False,
        )[0]
        self.premise_representations = get_premise_representations(sim=sim_premise_to_all_claims, softmax=softmax)
        self.ratio = cluster_ratio

    def rank(self, claim_id: int, premise_ids: Sequence[str], k: int) -> Sequence[str]:  # noqa: D102
        return _premise_cluster_filtered(
            premise_ids=premise_ids,
            premise_repr=torch.stack(
                [
                    self.premise_representations[pid]
                    for pid in premise_ids
                ],
                dim=0,
            ),
            k=k,
            ratio=self.ratio,
            similarity_lookup=self.similarity_lookup(for_claim_id=claim_id),
        )
