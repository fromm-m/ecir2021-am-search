import pathlib
import random
import string
import tempfile
import unittest
from typing import Any, List, Mapping, MutableMapping, Optional, Sequence, Type

import torch

from arclus.models import get_baseline_method_by_name
from arclus.models.base import RankingMethod
from arclus.models.baselines import ZeroShotClusterKNN, ZeroShotKNN, core_set, get_premise_representations, get_query_claim_similarities
from arclus.similarity import LpSimilarity


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
        return kwargs

    def test_rank(self):
        """Test rank."""
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
