import pandas as pd
import torch

from arclus.negative_sampling import NegativeSampler
from arclus.settings import NEGATIVE_SAMPLES, NUM_NEG_PER_POS, PREP_ASSIGNMENTS, PREP_CLAIMS, PREP_PREMISES


def main():
    # the true samples, a tensor of shape (n, 2)
    assignments = pd.read_csv(PREP_ASSIGNMENTS)
    premises = pd.read_csv(PREP_PREMISES)
    claims = pd.read_csv(PREP_CLAIMS)
    true_samples = torch.from_numpy(assignments[['claim_id', 'premise_id']].values)

    premises = premises[["premise_text"]].values
    claims = claims[["claim_text"]].values

    # We can increase this. If batch_size = num_samples, we have one a single batch. This may have large memory requirements
    batch_size = 1024
    # Instantiate the sampler
    sampler = NegativeSampler(num_premises=len(premises), num_claims=len(claims), num_neg_per_pos=NUM_NEG_PER_POS)
    columns = ["claim_id", "premise_id"]
    negative_samples = pd.DataFrame(columns=columns)

    # Generate negative samples
    for batch in true_samples.split(batch_size):
        # negative batch contains the IDs of claim / premise pairs.
        corrupt_premises = sampler.sample(batch=batch, corrupt_premises=True)  # lookup strings
        neg_batch = torch.stack([
            batch[:, 0].view(-1, 1).repeat(1, corrupt_premises.shape[1]),
            corrupt_premises
        ], dim=-1)
        # view as a large batch
        neg_batch = neg_batch.view(-1, 2)
        # negative_batch_str = [
        #    claims[claim_id] + ' ||| ' + premises[premise_id]
        #    for claim_id, premise_id in neg_batch.tolist()
        # ]
        temp_df = pd.DataFrame(neg_batch.tolist(), columns=columns)
        negative_samples = negative_samples.append(temp_df)
    negative_samples.to_csv(NEGATIVE_SAMPLES)


if __name__ == '__main__':
    main()
