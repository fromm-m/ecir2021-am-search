import unittest

import torch

from arclus.negative_sampling import NegativeSampler


def _verify_negative_samples(
    batch: torch.LongTensor,
    corrupted: torch.LongTensor,
    num_neg_per_pos: int,
    num_ids: int,
    col: int,
) -> None:
    # check type
    assert corrupted.dtype == torch.long

    # check shape
    batch_size = batch.shape[0]
    assert corrupted.shape == (batch_size, num_neg_per_pos)

    # check value range
    assert (0 <= corrupted).all()
    assert (corrupted < num_ids).all()

    # check negativity
    assert (batch[:, col, None] != corrupted).all()


class NegativeSamplerTest(unittest.TestCase):
    """unittest for NegativeSampler."""

    #: The number of premises
    num_premises: int = 7

    #: The number of claims
    num_claims: int = 5

    #: The number of negative samples per positive one
    num_neg_per_pos: int = 2

    #: The batch size
    batch_size: int = 3

    #: The instance to test
    instance: NegativeSampler

    #: The batch to test on
    batch: torch.LongTensor

    def setUp(self) -> None:
        """Setup the instance and a test batch."""
        self.instance = NegativeSampler(
            num_premises=self.num_premises,
            num_claims=self.num_claims,
            num_neg_per_pos=self.num_neg_per_pos,
        )
        self.batch = torch.stack([
            torch.randint(num, size=(self.batch_size,))
            for num in (self.num_claims, self.num_premises)
        ], dim=1)

    def test_sample_premises(self):
        """unittest for sample(..., corrupt_premises=True)."""
        corrupted_premises = self.instance.sample(
            batch=self.batch,
            corrupt_premises=True,
        )

        _verify_negative_samples(
            batch=self.batch,
            corrupted=corrupted_premises,
            num_neg_per_pos=self.num_neg_per_pos,
            num_ids=self.num_premises,
            col=1,
        )

    def test_sample_claims(self):
        """unittest for sample(..., corrupt_premises=False)."""
        corrupted_claims = self.instance.sample(
            batch=self.batch,
            corrupt_premises=False,
        )

        _verify_negative_samples(
            batch=self.batch,
            corrupted=corrupted_claims,
            num_neg_per_pos=self.num_neg_per_pos,
            num_ids=self.num_claims,
            col=0,
        )
