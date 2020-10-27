import pathlib
import random
import string
import tempfile
import unittest
from typing import Any, Collection, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Type

import numpy
import pandas
import torch

from arclus.models import LearnedSimilarityClusterKNN, get_baseline_method_by_name
from arclus.models.base import RankingMethod
from arclus.models.learned_similarity import BiasedCoreset, Coreset, LearnedSimilarityKNN, core_set, get_claim_similarity_premise_representations, get_query_claim_similarities
from arclus.models.zero_shot import ZeroShotClusterKNN, ZeroShotKNN
from arclus.settings import PREP_TEST_PRODUCT_SIMILARITIES, PREP_TEST_SIMILARITIES, PREP_TEST_STATES
from arclus.similarity import LpSimilarity


def _generate_random_data(
    all_claim_ids: Sequence[int],
    premise_ids: Sequence[str],
    max_premises: int,
    min_premises: int,
    num_clusters: int,
) -> pandas.DataFrame:
    data = []
    clusters = string.ascii_letters[:num_clusters]
    for claim in all_claim_ids:
        num = random.randrange(min_premises, max_premises)
        premises = random.sample(premise_ids, num)
        relevance = numpy.random.randint(3, size=num)
        this_clusters = [random.choice(clusters) for _ in range(num)]
        claims = [claim] * num
        data.extend(zip(claims, premises, relevance, this_clusters))
    return pandas.DataFrame(
        data=data,
        columns=[
            "claim_id",
            "premise_id",
            "relevance",
            "premiseClusterID_groundTruth",
        ])


class RankingTests:
    """Base class for ranking tests."""
    num_premises: int = 17
    num_claims: int = 13
    num_queries: int = 7
    k: int = 5
    dim: int = 3
    cls: Type[RankingMethod]
    kwargs: Optional[Mapping[str, Any]] = None
    instance: RankingMethod
    all_premise_ids: List[str]

    def setUp(self):
        """Instantiate test instance."""
        self.kwargs = self.kwargs or {}
        self.kwargs = self._pre_instantiation_hook(kwargs=self.kwargs)
        self.instance = self.cls(**self.kwargs)

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """Prepare instantiation."""
        self.all_premise_ids = [c for c in string.ascii_letters[:self.num_premises]]
        self.all_query_claim_ids = list(range(self.num_claims))
        self.all_result_claim_ids = list(range(self.num_claims, 3 * self.num_claims))
        num_clusters = 3
        min_premises = self.k
        max_premises = self.num_premises // 2
        self.training_data = _generate_random_data(
            all_claim_ids=self.all_query_claim_ids,
            premise_ids=self.all_premise_ids,
            max_premises=max_premises,
            min_premises=min_premises,
            num_clusters=num_clusters,
        )
        return kwargs

    def test_fit(self):
        """Test fit."""
        self.instance.fit(
            training_data=self.training_data,
            k=self.k,
        )

    def test_rank(self):
        """Test rank."""
        self.instance.fit(training_data=self.training_data, k=self.k)
        premise_ids = self.training_data.loc[self.training_data["claim_id"] == self.all_query_claim_ids[0]]["premise_id"].tolist()
        assert len(premise_ids) >= self.k
        ranking = self.instance.rank(claim_id=0, premise_ids=premise_ids, k=self.k)
        assert isinstance(ranking, (list, tuple))
        assert len(ranking) == self.k
        assert set(ranking).issubset(premise_ids)

    def test_get_baseline_method_by_name(self):
        """Test get_baseline_method_by_name."""
        method = get_baseline_method_by_name(name=self.cls.__name__, **self.kwargs)
        assert method.__class__ is self.cls


class ZeroShotTests(RankingTests):
    """Base class for zero-shot ranking tests."""

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)

        self.tmp_dir = tempfile.TemporaryDirectory()
        tmp_path = pathlib.Path(self.tmp_dir.name)

        # save dummy claim representations
        claims_path = tmp_path / 'claims.pt'
        claims = {
            claim_id: torch.rand(self.dim)
            for claim_id in range(self.num_claims)
        }
        torch.save(claims, claims_path)
        kwargs['claims_path'] = claims_path

        # save dummy premise representations
        premises_path = tmp_path / 'premises.pt'
        premises = {
            premise_id: torch.rand(self.dim)
            for premise_id in self.all_premise_ids
        }
        torch.save(premises, premises_path)
        kwargs['premises_path'] = premises_path

        return kwargs

    def tearDown(self):
        """Cleanup files."""
        self.tmp_dir.cleanup()


class ZeroShotKNNTests(ZeroShotTests, unittest.TestCase):
    """Tests for ZeroShotKNN."""

    cls = ZeroShotKNN
    kwargs = dict(
        similarities=["l2", "cos"],
    )


class ZeroShotClusterKNNTests(ZeroShotTests, unittest.TestCase):
    """Tests for ZeroShotClusterKNN."""

    cls = ZeroShotClusterKNN
    kwargs = dict(
        cluster_ratios=[0.1, 0.5],
        similarities=["l1", "cos"],
    )


def test_get_query_claim_similarities():
    """Test get_query_claim_similarities."""
    num_premises = 7
    num_claims = 5
    # sparse information, i.e. not every claim-premise pair has precomputed similarities
    drop_prob = .5
    old_sim = {
        (premise_id, claim_id): 10.0 * torch.rand(2, )
        for premise_id in map(str, range(num_premises))
        for claim_id in range(num_claims)
        if random.random() > drop_prob
    }
    for softmax in (False, True):
        sim = get_query_claim_similarities(
            sim=old_sim,
            softmax=softmax,
        )
        assert set(sim.keys()) == set(old_sim.keys())
        for v in sim.values():
            assert torch.is_tensor(v)
            assert v.dtype == torch.float32
            # scalar
            assert v.numel() == 1
            if softmax:
                assert 0 <= v.item() <= 1


def test_get_premise_representations():
    """Test get_premise_representations."""
    num_premises = 7
    num_claims = 5
    premise_ids = list(map(str, range(num_premises)))
    claim_ids = range(num_claims)
    old_sim = {
        (premise_id, claim_id): 10.0 * torch.rand(1, )
        for premise_id in premise_ids
        for claim_id in claim_ids
    }
    premise_to_query_claim = old_sim.keys()
    sim = get_claim_similarity_premise_representations(
        sim=old_sim,
        premise_to_query_claim=premise_to_query_claim,
    )
    assert set(sim.keys()) == set(premise_to_query_claim)
    for v in sim.values():
        assert torch.is_tensor(v)
        assert v.dtype == torch.float32
        assert v.shape == (num_claims,)


class CoreSetUtilityTests(unittest.TestCase):
    """Tests for core_set method."""

    def test_randomized(self):
        """Randomized test."""
        n = 10
        similarity = torch.rand(n, n)
        first_id = random.randrange(n)
        k = 3
        result = core_set(similarity=similarity, k=k)
        assert isinstance(result, Sequence)
        assert len(result) == k
        assert len(set(result)) == len(result)
        assert all(0 <= i < n for i in result)
        assert result[0] == first_id

    def test_manual(self):
        """Hand-curated test."""
        vectors = torch.as_tensor(
            data=[
                [0, 0],
                [1, 0],
                [2, 0],
                [1, 1],
            ],
            dtype=torch.float32,
        )
        similarity = LpSimilarity(p=2).sim(vectors, vectors)
        result = core_set(similarity=similarity, k=3)
        assert result == [0, 2, 3]


def _generate_random_product_similarities(
    premise_ids: Collection[str],
    result_claim_ids: Collection[int]
) -> Mapping[Tuple[str, int], torch.FloatTensor]:
    return {
        (pid, cid): torch.rand(2, )
        for pid in premise_ids
        for cid in result_claim_ids
    }


def _generate_random_pair_similarities(premise_claim_pairs: Collection[Tuple[str, int]]) -> Mapping[Tuple[str, int], torch.FloatTensor]:
    return {
        (pid, cid): torch.rand(2, )
        for pid, cid in premise_claim_pairs
    }


def _generate_random_premise_states(premise_claim_pairs: Collection[Tuple[str, int]], dim: int = 3) -> Mapping[Tuple[str, int], torch.FloatTensor]:
    return {
        (pid, cid): torch.rand(dim, )
        for pid, cid in premise_claim_pairs
    }


class PrecomputedSimilarityDependentTests(RankingTests):
    """Base class for ranking methods dependent on precomputed similarities."""

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)

        self.tmp_dir = tempfile.TemporaryDirectory()
        tmp_path = pathlib.Path(self.tmp_dir.name)

        similarities_dir = tmp_path
        model_name = "abc123"
        model_path = tmp_path / model_name
        similarities_dir2 = similarities_dir / model_name
        similarities_dir2.mkdir(exist_ok=True, parents=True)

        # create dummy precomputed similarities
        product_path = similarities_dir2 / PREP_TEST_PRODUCT_SIMILARITIES
        torch.save(_generate_random_product_similarities(premise_ids=self.all_premise_ids, result_claim_ids=self.all_result_claim_ids), product_path)
        premise_query_claim_pairs = set(zip(*[self.training_data[col] for col in ["premise_id", "claim_id"]]))
        pair_path = similarities_dir2 / PREP_TEST_SIMILARITIES
        torch.save(_generate_random_pair_similarities(premise_claim_pairs=premise_query_claim_pairs), pair_path)
        state_path = similarities_dir2 / PREP_TEST_STATES
        torch.save(_generate_random_premise_states(premise_claim_pairs=premise_query_claim_pairs), state_path)

        kwargs["similarities_dir"] = tmp_path
        kwargs["model_path"] = model_path

        return kwargs

    def tearDown(self):
        """Cleanup files."""
        self.tmp_dir.cleanup()


class CoreSetTests(PrecomputedSimilarityDependentTests, unittest.TestCase):
    """Test for CoreSet ranking method."""

    cls = Coreset
    kwargs = dict(
        premise_premise_similarities=["cos", "l1"],
    )


class BiasedCoreSetTests(PrecomputedSimilarityDependentTests, unittest.TestCase):
    """Test for BiasedCoreSet ranking method."""

    cls = BiasedCoreset
    kwargs = dict(
        premise_premise_similarities=["l2", "l1"],
        resolution=10,
    )


class LearnedSimilarityKNNTests(PrecomputedSimilarityDependentTests, unittest.TestCase):
    """Test for LearnedSimilarityKNN ranking method."""

    cls = LearnedSimilarityKNN
    kwargs = dict(
        softmax=True,
    )


class LearnedSimilarityClusterKNNTests(PrecomputedSimilarityDependentTests, unittest.TestCase):
    """Test for LearnedSimilarityClusterKNN ranking method."""

    cls = LearnedSimilarityClusterKNN
    kwargs = dict(
        softmax=True,
        cluster_ratios=[0.2, 0.4],
    )
