"""Tests for evaluation methods."""
import numpy

from arclus.evaluation import mdcg


def test_mdcg():
    """Tests for modified DCG."""

    # settings
    num_premises = 17
    num_claims = 13
    num_clusters = 7
    num_queries = 5
    k = 3

    # generate random data
    query_claim_ids = numpy.random.randint(num_claims, size=(num_queries,))
    predicted_rankings = numpy.stack([numpy.random.permutation(num_premises)[:k] for _ in range(num_queries)])
    premise_cluster_ids = numpy.random.randint(num_clusters, size=(num_premises,))
    premise_relevance = numpy.random.randint(3, size=(num_claims, num_premises)).astype(numpy.float64)

    scores = mdcg(
        query_claim_ids=query_claim_ids,
        predicted_rankings=predicted_rankings,
        premise_cluster_ids=premise_cluster_ids,
        premise_relevance=premise_relevance,
    )

    assert scores.dtype == numpy.float64
    assert scores.shape == (num_queries,)

    # non-negative scores
    numpy.testing.assert_array_less(0, scores)
