import torch


class NegativeSampler:
    """Base class for negative sampling."""

    def __init__(
        self,
        max_premise_id: int,
        max_claim_id: int,
        num_neg_per_pos: int = 1,
    ):
        super().__init__()
        self.max_ids = (max_claim_id, max_premise_id)
        self.num_neg_per_pos = num_neg_per_pos

    def sample(self, batch: torch.LongTensor, corrupt_premises: bool = True) -> torch.LongTensor:
        """
        Sample negatives for a given batch of positives.

        :param batch: shape: (batch_size, 2)
            The positive samples, each row contains (claim_id, premise_id).
        :param corrupt_premises:
            Whether to corrupt premises, or claims.

        :return: shape: (batch_size, num_neg)
            The negative samples, i.e. (batch[b, col], result[b, i]) is a negative sample, where col is 0 if
            corrupt_premises else 1.
        """
        col = 0 if corrupt_premises else 1
        batch_size = batch.shape[0]
        corrupted = torch.randint(self.max_ids[col] - 1, size=(batch_size, self.num_neg_per_pos))
        corrupted += (corrupted >= batch[:, col]).long()
        return corrupted
