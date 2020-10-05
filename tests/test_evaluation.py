"""Tests for evaluation methods."""
import math
import unittest

import numpy
import pandas

from arclus.evaluation import mndcg_score, optimal_mdcg_score
from arclus.utils import generate_random_words


class MNDCGTests(unittest.TestCase):
    """Tests for computation of modified NDCG score."""

    num_premises: int = 13
    k: int = 5
    num_clusters: int = 7

    def setUp(self) -> None:
        """Initialize test data."""
        # random data
        self.data = pandas.DataFrame(data=dict(
            premise_id=generate_random_words(num_words=self.num_premises),
            premiseClusterID_groundTruth=numpy.random.randint(self.num_clusters, size=(self.num_premises,)),
            relevance=numpy.random.randint(3, size=(self.num_premises,)),
        ))
        self.data['premiseClusterID_groundTruth'] = self.data['premiseClusterID_groundTruth'].astype(str)

        # random ranking
        self.y_pred = self.data['premise_id'].values.copy()
        numpy.random.shuffle(self.y_pred)

    def test_mndcg_score(self):
        """Test mndcg_score."""
        score = mndcg_score(
            y_pred=self.y_pred,
            data=self.data,
            k=self.k,
        )

        # check type
        assert isinstance(score, float)

        # check value range
        assert 0 <= score <= 1.

    def test_optimal_score_manual(self):
        """Manually test optimal_score."""
        test_data = pandas.DataFrame(
            data=dict(
                relevance=[
                    0, 1, 1, 1, 2, 2, 2, 0,
                ],
                premiseClusterID_groundTruth=[
                    'a', 'a', 'a', 'b', 'b', 'c', 'c', 'z',
                ]
            )
        )
        # optimal ranking: (c, 2), (b, 2), (a, 1)
        # expected_score
        expected_score = (numpy.asarray([2, 2, 1], dtype=float) / numpy.log2(2 + numpy.arange(3))).sum()
        optimal_score = optimal_mdcg_score(
            data=test_data,
            k=3,
        )
        self.assertAlmostEqual(expected_score, optimal_score)
