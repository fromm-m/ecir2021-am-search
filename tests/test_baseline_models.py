import pathlib
import random
import string
import tempfile
import unittest
from collections import defaultdict
from typing import Any, List, Mapping, MutableMapping, Optional, Sequence, Type
from unittest.case import SkipTest

import numpy
import pandas
import torch

from arclus.models import get_baseline_method_by_name
from arclus.models.base import RankingMethod
from arclus.models.baselines import BiasedCoreset, Coreset, ZeroShotClusterKNN, ZeroShotKNN, core_set, get_premise_representations, get_query_claim_similarities
from arclus.similarity import CosineSimilarity, LpSimilarity


def _generate_random_data(
    num_claims: int,
    premise_ids: Sequence[str],
    max_premises: int,
    min_premises: int,
    num_clusters: int,
) -> pandas.DataFrame:
    data = []
    clusters = string.ascii_letters[:num_clusters]
    for claim in range(num_claims):
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
        num_clusters = 3
        min_premises = 3
        max_premises = self.num_premises // 2
        self.training_data = _generate_random_data(
            num_claims=self.num_claims,
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
        this_premise_ids = random.sample(self.all_premise_ids, 2 * self.k)
        ranking = self.instance.rank(claim_id=0, premise_ids=this_premise_ids, k=self.k)
        assert isinstance(ranking, (list, tuple))
        assert len(ranking) == self.k
        assert set(ranking).issubset(this_premise_ids)

    def test_get_baseline_method_by_name(self):
        """Test get_baseline_method_by_name."""
        method = get_baseline_method_by_name(name=self.cls.__name__, **self.kwargs)
        assert method.__class__ is self.cls


class ZeroShotTests(RankingTests):
    """Base class for zero-shot ranking tests."""

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs['similarity'] = LpSimilarity(p=1)

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


class ZeroShotClusterKNNTests(ZeroShotTests, unittest.TestCase):
    """Tests for ZeroShotClusterKNN."""

    cls = ZeroShotClusterKNN
    kwargs = dict(
        cluster_ratio=0.5,
    )


class ZeroShotClusterKNNTests2(ZeroShotTests, unittest.TestCase):
    """Tests for ZeroShotClusterKNN."""

    cls = ZeroShotClusterKNN
    kwargs = dict(
        cluster_ratio=1.0,
        cluster_representative='closest-to-claim',
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
    old_sim = {
        (premise_id, claim_id): 10.0 * torch.rand(2, )
        for premise_id in premise_ids
        for claim_id in range(num_claims)
    }
    for softmax in (False, True):
        sim = get_premise_representations(
            sim=old_sim,
            softmax=softmax,
        )
        assert set(sim.keys()) == set(premise_ids)
        for v in sim.values():
            assert torch.is_tensor(v)
            assert v.dtype == torch.float32
            assert v.shape == (num_claims,)
            if softmax:
                assert (v >= 0).all() and (v <= 1).all()


class CoreSetUtilityTests(unittest.TestCase):
    """Tests for core_set method."""

    def test_randomized(self):
        """Randomized test."""
        n = 10
        similarity = torch.rand(n, n)
        first_id = random.randrange(n)
        k = 3
        result = core_set(similarity=similarity, first_id=first_id, k=k)
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
        result = core_set(similarity=similarity, first_id=0, k=3)
        assert result == [0, 2, 3]


class PrecomputedSimilarityDependentTests(RankingTests):
    """Base class for ranking methods dependent on precomputed similarities."""

    def setUp(self):  # noqa: D102
        # mock
        self.kwargs = self.kwargs or {}
        self.kwargs = self._pre_instantiation_hook(kwargs=self.kwargs)
        self.instance = self.cls.__new__(self.cls)

        # mock
        fake_similarities = defaultdict(lambda: random.random())
        self.instance.__dict__["precomputed_similarities"] = fake_similarities

        fake_states = defaultdict(lambda: torch.rand(self.dim))
        self.instance.__dict__["precomputed_states"] = fake_states

        for key, value in self.kwargs.items():
            self.instance.__dict__[key] = value

        self._post_fake_instantiation_hook()

    def _post_fake_instantiation_hook(self):
        pass

    def test_get_baseline_method_by_name(self):
        raise SkipTest("Expensive initialization")


class CoreSetTests(PrecomputedSimilarityDependentTests, unittest.TestCase):
    """Test for CoreSet ranking method."""

    cls = Coreset
    kwargs = dict(
        premise_premise_similarity=CosineSimilarity(),
        debug=False,
    )

    def _post_fake_instantiation_hook(self):  # noqa: D102
        self.instance.__dict__["threshold"] = None


class BiasCoreSetTests(PrecomputedSimilarityDependentTests, unittest.TestCase):
    """Test for BiasedCoreSet ranking method."""

    cls = BiasedCoreset
    kwargs = dict(
        premise_premise_similarity=CosineSimilarity(),
        resolution=10,
        debug=False,
    )

    def _post_fake_instantiation_hook(self):  # noqa: D102
        self.instance.__dict__["alpha"] = None
