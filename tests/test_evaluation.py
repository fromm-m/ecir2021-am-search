"""Tests for evaluation methods."""
import unittest

import numpy

from arclus.evaluation import get_optimal_ranking, mdcg, nmdcg


class ModifiedNDCGTests(unittest.TestCase):
    """Tests for modified NDCG."""

    # settings
    num_premises = 17
    num_claims = 13
    num_clusters = 7
    num_queries = 5
    k = 3

    def setUp(self) -> None:
        # generate random data
        self.query_claim_ids = numpy.random.randint(self.num_claims, size=(self.num_queries,))
        self.predicted_rankings = numpy.stack([numpy.random.permutation(self.num_premises)[:self.k] for _ in range(self.num_queries)])
        self.premise_cluster_ids = numpy.random.randint(self.num_clusters, size=(self.num_premises,))
        self.premise_relevance = numpy.random.randint(3, size=(self.num_claims, self.num_premises)).astype(numpy.float64)

    def _common_verification(self, scores: numpy.ndarray):
        assert scores.dtype == numpy.float64
        assert scores.shape == (self.num_queries,)
        # non-negative scores
        assert (scores >= 0.0).all()

    def test_dcg(self):
        """Tests mDCG computation."""
        scores = mdcg(
            query_claim_ids=self.query_claim_ids,
            predicted_rankings=self.predicted_rankings,
            premise_cluster_ids=self.premise_cluster_ids,
            premise_relevance=self.premise_relevance,
        )
        self._common_verification(scores=scores)

    def test_get_optimal_ranking(self):
        """Test get_optimal_ranking."""
        optimal_ranking = get_optimal_ranking(
            query_claim_ids=self.query_claim_ids,
            premise_cluster_ids=self.premise_cluster_ids,
            premise_relevance=self.premise_relevance,
            k=self.k,
        )
        assert optimal_ranking.shape == (self.num_queries, self.k)
        assert optimal_ranking.dtype == numpy.int64
        # check value range
        numpy.testing.assert_array_less(-1, optimal_ranking)
        numpy.testing.assert_array_less(optimal_ranking, self.num_premises)
        # check unique
        for r in optimal_ranking:
            assert numpy.unique(r).shape == r.shape

    def test_get_optimal_ranking_manual(self):
        """Test get_optimal_ranking on a manually computed example."""
        premise_relevance = numpy.array([
            [0, 1, 2, 3, 4],
            [3, 2, 1, 4, 0],
        ], dtype=numpy.float64)
        premise_cluster_ids = numpy.array([0, 0, 1, 1, 1])
        expected_optimal_ranking = numpy.array([
            [4, 1],
            [3, 0],
        ])
        query_claim_ids = numpy.arange(premise_relevance.shape[0])
        optimal_ranking = get_optimal_ranking(
            query_claim_ids=query_claim_ids,
            premise_cluster_ids=premise_cluster_ids,
            premise_relevance=premise_relevance,
            k=2,
        )
        numpy.testing.assert_equal(expected_optimal_ranking, optimal_ranking)

    def test_nmdcg(self):
        """Test NmDCG computation."""
        scores = nmdcg(
            query_claim_ids=self.query_claim_ids,
            predicted_rankings=self.predicted_rankings,
            premise_cluster_ids=self.premise_cluster_ids,
            premise_relevance=self.premise_relevance,
        )
        self._common_verification(scores=scores)
        # check value range
        assert (scores <= 1.0).all()
