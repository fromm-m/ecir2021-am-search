import torch


class NegativeSampler:
    """Base class for negative sampling."""

    def __init__(
        self,
        num_premises: int,
        num_claims: int,
        num_neg_per_pos: int = 1,
    ):
        super().__init__()
        self.num = (num_claims, num_premises)
        self.num_neg_per_pos = num_neg_per_pos

    def sample(self, batch: torch.LongTensor, corrupt_premises: bool = True) -> torch.LongTensor:
        """
        Sample negatives for a given batch of positives.

        :param batch: shape: (batch_size, 2)
            The positive samples, each row contains (claim_id, premise_id).
        :param corrupt_premises:
            Whether to corrupt premises, or claims.

        :return: shape: (batch_size, num_neg)
            The negative samples, i.e. (batch[b, col], result[b, i]) is a negative sample, where col is 1 if
            corrupt_premises else 0.
        """
        col = 1 if corrupt_premises else 0
        batch_size = batch.shape[0]
        corrupted = torch.randint(self.num[col] - 1, size=(batch_size, self.num_neg_per_pos))
        corrupted += (corrupted >= batch[:, col, None]).long()
        return corrupted
