import enum
import logging
import pathlib
from abc import ABC
from collections import defaultdict
from operator import itemgetter
from typing import Any, Callable, Collection, Mapping, Optional, Sequence, Tuple, Union

import numpy
import pandas
import torch
from sklearn.cluster import KMeans

from .base import RankingMethod
from .zero_shot import _average_scores
from ..evaluation import mndcg_score
from ..settings import PREMISES_TEST_FEATURES, PREP_ASSIGNMENTS_TEST, PREP_TEST_PRODUCT_SIMILARITIES, PREP_TEST_SIMILARITIES, PREP_TEST_STATES
from ..similarity import CosineSimilarity, Similarity, get_similarity_by_name
from ..utils import resolve_num_clusters
from ..utils_am import inference_no_args, load_bert_model_and_data_no_args

logger = logging.getLogger(__name__)


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


def get_claim_similarity_premise_representations(
    sim: Mapping[Tuple[str, int], Union[torch.FloatTensor, float]],
    premise_to_query_claim: Collection[Tuple[str, int]],
) -> Mapping[Tuple[str, int], torch.FloatTensor]:
    """
    Construct premise representations as similarity vectors to claims.

    :param sim:
        The similarities for (premise_id, claim_id) pairs.
    :param premise_to_query_claim:
        A mapping from premise IDs to the query claim ID.

    :return:
        A mapping from (claim_id, premise_id) to a vector of similarities, shape: (num_claims,).
    """
    # verify that similarities are available for all claim, premise pairs
    premise_ids, result_claim_ids = [
        sorted(set(map(itemgetter(pos), sim.keys())))
        for pos in (0, 1)
    ]

    assert set(sim.keys()) == set(
        (pid, cid)
        for pid in premise_ids
        for cid in result_claim_ids
    )

    # convert to tensor, shape: (num_premises, num_resultclaims)
    sim = torch.as_tensor(
        data=[
            [
                float(sim[premise_id, result_claim_id])
                for result_claim_id in result_claim_ids
            ]
            for premise_id in premise_ids
        ],
        dtype=torch.float32,
    )
    assert sim.shape == (len(premise_ids), len(result_claim_ids))

    # one row corresponds to one premise representation
    premise_repr = dict(zip(premise_ids, sim))

    # add query claim to key
    result = {
        (pid, cid): premise_repr[pid]
        for pid, cid in premise_to_query_claim
    }

    return result


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
    algorithm = KMeans(n_clusters=resolve_num_clusters(ratio=ratio, num_premises=num_premises, k=k))
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
            product=product
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


def get_premise_representations_for_claim(
    claim_id: int,
    premise_ids: Sequence[str],
    source: Mapping[Tuple[str, int], torch.FloatTensor],
) -> torch.FloatTensor:
    """
    Lookup premise representations from a dictionary of precomputed representations.

    :param claim_id:
        The claim ID.
    :param premise_ids:
        The premise IDs.
    :param source:
        The precomputed representations, all vectors of same shape.

    :return: shape: (num_premises, dim)
        A tensor of premise representations.
    """
    """Get premise representations."""
    return torch.stack(
        [
            source[premise_id, claim_id]
            for premise_id in premise_ids
        ],
        dim=0,
    )


class PremiseRepresentationEnum(str, enum.Enum):
    """Enum for premise representation options."""

    #: Do not use premise representations
    none = "none"

    #: Use pre-trained BERT representations
    zero_shot_bert = "zero_shot_bert"

    #: Use the hidden states of the learned similarity model.
    learned_similarity_last_layer = "learned_similarity_last_layer"

    #: Use a vector of claim similarities
    learned_similarity_claim_similarities = "learned_similarity_claim_similarities"


class LearnedSimilarityBasedMethod(RankingMethod, ABC):
    """Base class for ranking methods based on learned similarity between claims and premises."""

    #: The precomputed similarities.
    precomputed_similarities: Mapping[Tuple[str, int], float]

    #: The precomputed representations
    premise_representations: Optional[Mapping[Tuple[str, int], torch.FloatTensor]]

    def __init__(
        self,
        softmax: bool,
        model_path: str,
        similarities_dir: str,
        cache_root: Optional[str] = None,
        premise_representation: PremiseRepresentationEnum = PremiseRepresentationEnum.learned_similarity_last_layer,
        premise_representation_kwargs: Optional[Mapping[str, Any]] = None,
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
        self.precomputed_similarities, self.premise_representations = _load_or_compute_similarities(
            cache_root=cache_root,
            model_path=model_path,
            similarities_dir=similarities_dir,
            softmax=softmax,
            product=False,
            with_states=premise_representation == PremiseRepresentationEnum.learned_similarity_last_layer,
        )
        if premise_representation_kwargs is None:
            premise_representation_kwargs = dict()
        if premise_representation == PremiseRepresentationEnum.learned_similarity_claim_similarities:
            self.premise_representations = get_claim_similarity_premise_representations(
                sim=_load_or_compute_similarities(
                    cache_root=cache_root,
                    model_path=model_path,
                    similarities_dir=similarities_dir,
                    softmax=softmax,
                    product=True,
                    with_states=False,
                )[0],
                premise_to_query_claim=self.precomputed_similarities.keys(),
            )
        elif premise_representation == PremiseRepresentationEnum.zero_shot_bert:
            self.premises_representations = torch.load(PREMISES_TEST_FEATURES)
        elif premise_representation == PremiseRepresentationEnum.learned_similarity_last_layer:
            assert self.premise_representations is not None
        elif premise_representation == PremiseRepresentationEnum.none:
            assert self.premise_representations is None
        else:
            raise NotImplementedError

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
        cluster_ratios: Sequence[float],
        softmax: bool,
        model_path: str,
        similarities_dir: str,
        cache_root: Optional[str] = None,
        premise_representations: PremiseRepresentationEnum = PremiseRepresentationEnum.learned_similarity_last_layer,
    ):
        """
        Initialize the method.

        :param cluster_ratios: >0
            The relative number of clusters to use. If None, use k.
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
            premise_representation=premise_representations,
        )
        self.ratio = None
        self.ratios = cluster_ratios

    def fit(
        self,
        training_data: pandas.DataFrame,
        k: int,
    ) -> "RankingMethod":
        # TODO: Merge with ZeroShotCluster
        scores = defaultdict(list)
        for query_claim_id, group in training_data.groupby(by="claim_id"):
            premise_ids = group["premise_id"].tolist()
            premise_repr = get_premise_representations_for_claim(
                claim_id=query_claim_id,
                premise_ids=premise_ids,
                source=self.premise_representations,
            )
            similarity_lookup = self.similarity_lookup(for_claim_id=query_claim_id)
            for ratio in self.ratios:
                scores[ratio].append(mndcg_score(
                    y_pred=_premise_cluster_filtered(
                        premise_ids=premise_ids,
                        premise_repr=premise_repr,
                        k=k,
                        ratio=ratio,
                        similarity_lookup=similarity_lookup,
                    ),
                    data=group,
                    k=k,
                ))
        # average over claims
        scores = _average_scores(scores=scores, num=len(training_data["claim_id"].unique()))
        self.ratio = max(scores.items(), key=itemgetter(1))[0]
        return self

    def rank(self, claim_id: int, premise_ids: Sequence[str], k: int) -> Sequence[str]:  # noqa: D102
        if self.ratio is None:
            raise ValueError(f"{self.__class__.__name__} must be fit before rank is called.")

        premise_repr = get_premise_representations_for_claim(
            claim_id=claim_id,
            premise_ids=premise_ids,
            source=self.premise_representations,
        )
        similarity_lookup = self.similarity_lookup(for_claim_id=claim_id)
        return _premise_cluster_filtered(
            premise_ids=premise_ids,
            premise_repr=premise_repr,
            k=k,
            ratio=self.ratio,
            similarity_lookup=similarity_lookup,
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


class BaseCoreSetRanking(LearnedSimilarityBasedMethod, ABC):
    """Base class for core set approaches."""

    def __init__(
        self,
        model_path: str,
        similarities_dir: str,
        premise_premise_similarity: Similarity = CosineSimilarity(),
        cache_root: Optional[str] = None,
        debug: bool = False,
        premise_representation: PremiseRepresentationEnum = PremiseRepresentationEnum.learned_similarity_last_layer,
    ):
        """
        Initialize the method.

        :param model_path:
            Directory where the fine-tuned bert similarity model checkpoint is located.
        :param premise_premise_similarity:
            The similarity to use between premise representations.
        :param cache_root:
            The directory where temporary BERT inference files are stored.
        :param debug:
            Whether to store fit artifacts for further inspection.
        """
        super().__init__(
            model_path=model_path,
            cache_root=cache_root,
            softmax=True,
            similarities_dir=similarities_dir,
            premise_representation=premise_representation,
        )
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
        premise_repr = get_premise_representations_for_claim(
            claim_id=claim_id,
            premise_ids=premise_ids,
            source=self.premise_representations,
        )

        # compute pair-wise similarity matrix
        similarity = self.premise_premise_similarity.sim(left=premise_repr, right=premise_repr)

        return first_id, similarity


class Coreset(BaseCoreSetRanking):
    """Basic core-set approach with thresholding."""

    def __init__(
        self,
        model_path: str,
        similarities_dir: str,
        premise_premise_similarity: Similarity = CosineSimilarity(),
        cache_root: Optional[str] = None,
        debug: bool = False,
        fill_to_k: bool = False,
        premise_representation: PremiseRepresentationEnum = PremiseRepresentationEnum.learned_similarity_last_layer,
    ):
        """
        Initialize the method.

        :param model_path:
            Directory where the fine-tuned bert similarity model checkpoint is located.
        :param cache_root:
            The directory where temporary BERT inference files are stored.
        :param fill_to_k:
            Whether to fill up with more candidates (according to KNN heuristic), if less than k candidates remain
            after thresholding.
        """
        super().__init__(
            model_path=model_path,
            similarities_dir=similarities_dir,
            premise_premise_similarity=premise_premise_similarity,
            cache_root=cache_root,
            debug=debug,
            premise_representation=premise_representation,
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
            numpy.save(f"/tmp/scores_k{k}_{self.premise_premise_similarity}_{fold_hash}.npy",
                       numpy.stack([thresholds, scores]))
            self.threshold = thresholds[scores.argmax()]
        else:
            self.threshold = max(thresholds, key=_eval_threshold)

    def rank(self, claim_id: int, premise_ids: Sequence[str], k: int) -> Sequence[str]:  # noqa: D102
        if self.threshold is None:
            raise ValueError(f"{self.__class__.__name__} must be fit before rank is called.")

        return self._rank(
            claim_id=claim_id,
            premise_ids=premise_ids,
            k=k,
            threshold=self.threshold,
            fill_to_k=self.fill_to_k,
        )

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
        premise_representation: PremiseRepresentationEnum = PremiseRepresentationEnum.learned_similarity_last_layer,
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
            premise_representation=premise_representation,
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
