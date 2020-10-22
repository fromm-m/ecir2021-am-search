import torch
import pandas as pd
import numpy as np
from arclus.settings import PREMISES_FEATURES, PREP_ASSIGNMENTS_ENERGY, PREP_CLAIMS_ENERGY, \
    PREP_PREMISES_ENERGY, TRAIN_SIZE, TRAIN_PATH_ENERGY, TEST_PATH_ENERGY, VALID_PATH_ENERGY
import tqdm


def main():
    representations = torch.load(PREMISES_FEATURES)
    top_sim, top_id, sorted_premises = generate_mapping(representations)
    columns = ["premise_id", "claim_id"]
    positives = pd.read_csv(PREP_ASSIGNMENTS_ENERGY)[columns]
    positives = positives.apply(pd.to_numeric, errors="ignore")
    claims = pd.read_csv(PREP_CLAIMS_ENERGY)
    premises = pd.read_csv(PREP_PREMISES_ENERGY)
    list = []
    gt_mapping = positives.set_index('premise_id').T.to_dict(orient='list')
    k = 2
    for index, source_premise_id in tqdm.tqdm(enumerate(sorted_premises)):
        top_ids = top_id[index]
        source_claim_id = gt_mapping[int(source_premise_id)][0]
        i = 0
        for id in top_ids:
            target_premise_id = [sorted_premises[id.item()]][0]
            if not ((positives["premise_id"] == int(target_premise_id)) & (
                positives["claim_id"] == source_claim_id)).any():
                list.append((target_premise_id, source_claim_id))
                i = i + 1
                if i == k:
                    break
    negatives = pd.DataFrame(list, columns=columns)
    negatives_claims = negatives.merge(claims, on="claim_id")
    negatives_claims = negatives_claims.apply(pd.to_numeric, errors="ignore")
    negatives_complete = negatives_claims.merge(premises, on="premise_id")
    negatives_complete["label"] = "unsimilar"
    negatives_complete = negatives_complete\
        .drop_duplicates(subset=["claim_id", "premise_id"])

    positives_claims = positives.merge(claims, on="claim_id")
    positives_complete = positives_claims.merge(premises, on="premise_id")
    positives_complete["label"] = "similar"

    positives_complete = positives_complete[["premise_id", "claim_id", "claim_text", "premise_text", "label"]]
    negative_complete = negatives_complete[["premise_id", "claim_id", "claim_text", "premise_text", "label"]]
    combined = pd.concat([positives_complete, negative_complete])
    combined = combined.drop_duplicates(subset=["claim_id", "premise_id", "premise_text", "claim_text"], keep="first")
    combined_filtered = combined[["premise_id", "claim_id", "claim_text", "premise_text", "label"]]

    combined_shuffled = combined_filtered.sample(frac=1)
    msk_1 = np.random.rand(len(combined_shuffled)) < TRAIN_SIZE
    train = combined_shuffled[msk_1]
    test_valid = combined_shuffled[~msk_1]
    msk_2 = np.random.rand(len(test_valid)) < TRAIN_SIZE
    test = test_valid[msk_2]
    valid = test_valid[~msk_2]

    train.to_csv(TRAIN_PATH_ENERGY)
    test.to_csv(TEST_PATH_ENERGY)
    valid.to_csv(VALID_PATH_ENERGY)


def generate_mapping(representations):
    # fix order (necessary, since dict does not guarantee order)
    sorted_premises = sorted(representations.keys())
    # one array, shape: (num_premises, d)
    x = torch.stack([representations[k] for k in sorted_premises], dim=0)
    # normalize length
    x = torch.nn.functional.normalize(x, dim=-1)
    # compute pairwise cosine similarity
    sim = x @ x.t()
    # TODO: If this gives a memory error, batch left side
    # get top-k IDs
    top_sim, top_id = sim.topk(k=1000, largest=True, dim=1)
    return top_sim, top_id, sorted_premises

def filter_premises():
    print("hello")

if __name__ == '__main__':
    main()
