import pandas as pd
import torch

from arclus.negative_sampling import NegativeSampler
from arclus.settings import PREP_ASSIGNMENTS, PREP_CLAIMS, PREP_PREMISES, NUM_NEG_PER_POS

# the true samples, a tensor of shape (n, 2)
assignments = pd.read_csv(PREP_ASSIGNMENTS)
premises = pd.read_csv(PREP_PREMISES)
claims = pd.read_csv(PREP_CLAIMS)
true_samples = torch.from_numpy(assignments[['claim_id', 'premise_id']].values)

# We can increase this. If batch_size = num_samples, we have one a single batch. This may have large memory requirements
batch_size = 1024
# Instantiate the sampler
sampler = NegativeSampler(num_premises=len(premises), num_claims=len(claims), num_neg_per_pos=NUM_NEG_PER_POS)
# Generate negative samples
for batch in true_samples.split(batch_size):
    # negative batch contains the IDs of claim / premise pairs.
    negative_batch = sampler.sample(batch=batch)
    # lookup strings
    negative_batch_str = [
        claims[claim_id] + ' ||| ' + premises[premise_id]
        for claim_id, premise_id in negative_batch.tolist()
    ]
    # encode via BERT
    batch_encodings = bert.encode(negative_batch_str)
