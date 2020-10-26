"""Tests for pairwise similarity functions."""
import unittest
from typing import Any, Mapping, Type

import torch

from arclus.similarity import CSLSSimilarity, CosineSimilarity, LpSimilarity, Similarity


class SimilarityTests:
    """Common tests for all similarities."""

    cls: Type[Similarity]
    instance: Similarity
    kwargs: Mapping[str, Any] = dict()

    dim: int = 3
    n_claims: int = 5
    n_premises: int = 7

    def setUp(self):
        """Setup test instance."""
        self.instance = self.cls(**self.kwargs)

    def test_sim(self):
        """Test sim()."""
        claims = torch.rand(self.n_claims, self.dim)
        premises = torch.rand(self.n_premises, self.dim)
        sim = self.instance.sim(left=claims, right=premises)
        assert torch.is_tensor(sim)
        assert sim.shape == (self.n_claims, self.n_premises)


class LpSimilarityTests(SimilarityTests, unittest.TestCase):
    """Tests for LpSimilarity."""

    cls = LpSimilarity


class CosineSimilarityTests(SimilarityTests, unittest.TestCase):
    """Tests for CosineSimilarity."""

    cls = CosineSimilarity


class CSLSSimilarityTests(SimilarityTests, unittest.TestCase):
    """Tests for CSLSSimilarity."""

    cls = CSLSSimilarity
    kwargs = dict(
        base=CosineSimilarity(),
    )
