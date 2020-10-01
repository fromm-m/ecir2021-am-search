import pathlib
import tempfile
import unittest
from typing import Any, Mapping, MutableMapping, Optional, Type

import numpy
import torch

from arclus.get_similar import LpSimilarity
from arclus.models.baselines import RankingMethod, ZeroShotClusterKNN, ZeroShotKNN


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

    def setUp(self):
        """Instantiate test instance."""
        self.kwargs = self.kwargs or {}
        self.kwargs = self._pre_instantiation_hook(kwargs=self.kwargs)
        self.instance = self.cls(**self.kwargs)

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """Prepare instantiation."""
        return kwargs

    def test_rank(self):
        """Test rank."""
        queries = torch.randperm(self.num_claims)[:self.num_queries].tolist()
        ranking = self.instance.rank(queries=queries, k=self.k)
        assert isinstance(ranking, numpy.ndarray)
        assert ranking.shape == (self.num_queries, self.k)


class ZeroShotTests(RankingTests):
    """Base class for zero-shot ranking tests."""

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs['similarity'] = LpSimilarity(p=1)

        self.tmp_dir = tempfile.TemporaryDirectory()
        tmp_path = pathlib.Path(self.tmp_dir.name)

        # save dummy claim representations
        claims_path = tmp_path / 'claims.pt'
        claims = torch.rand(self.num_claims, self.dim)
        torch.save(claims, claims_path)
        kwargs['claims_path'] = claims_path

        # save dummy premise representations
        premises_path = tmp_path / 'premises.pt'
        premises = torch.rand(self.num_premises, self.dim)
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
