import unittest
from typing import Any, Mapping, Type

import torch

from arclus.get_similar import CSLSSimilarity, CosineSimilarity, LpSimilarity, Sim


class SimilarityTests:
    cls: Type[Sim]
    instance: Sim
    kwargs: Mapping[str, Any] = dict()

    dim: int = 3
    n_claims: int = 5
    n_premises: int = 7

    def setUp(self):
        self.instance = self.cls(**self.kwargs)

    def test_sim(self):
        claims = torch.rand(self.n_claims, self.dim)
        premises = torch.rand(self.n_premises, self.dim)
        sim = self.instance.sim(claims=claims, premises=premises)
        assert torch.is_tensor(sim)
        assert sim.shape == (self.n_claims, self.n_premises)


class LpSimilarityTests(SimilarityTests, unittest.TestCase):
    cls = LpSimilarity


class CosineSimilarityTests(SimilarityTests, unittest.TestCase):
    cls = CosineSimilarity


class CSLSSimilarityTests(SimilarityTests, unittest.TestCase):
    cls = CSLSSimilarity
    kwargs = dict(
        base=CosineSimilarity(),
    )
