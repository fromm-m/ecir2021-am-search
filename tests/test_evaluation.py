import unittest

import numpy
import pandas

from arclus.evaluation import mndcg_score
from arclus.utils import generate_random_words


class MNDCGTests(unittest.TestCase):
    """Tests for computation of mpodified NDCG score."""

    num_premises: int = 13
    k: int = 5
    num_clusters: int = 7

    def setUp(self) -> None:
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
