import pathlib
import random
import string
import tempfile
import unittest
from typing import Any, List, Mapping, MutableMapping, Optional, Type

import torch

from arclus.get_similar import LpSimilarity
from arclus.models.baselines import RankingMethod, ZeroShotClusterKNN, ZeroShotKNN, get_baseline_method_by_name


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
        n_clusters=10,
    )
