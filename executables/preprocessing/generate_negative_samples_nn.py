import torch
import pandas as pd
import numpy as np
from torch import nn
from arclus.settings import PREMISES_TEST_FEATURES_ENERGY, PREP_ASSIGNMENTS_ENERGY, PREP_CLAIMS_ENERGY, \
    PREP_PREMISES_ENERGY, TRAIN_SIZE, TRAIN_PATH_ENERGY, TEST_PATH_ENERGY, VALID_PATH_ENERGY
import tqdm
from pathlib import Path

def main():
    mapping = Path("mapping1.pt")
    if not mapping.is_file():
        generate_mapping1()
    mapping = torch.load(mapping)
    columns = ["premise_id", "claim_id"]
    positives = pd.read_csv(PREP_ASSIGNMENTS_ENERGY)[columns]
    claims = pd.read_csv(PREP_CLAIMS_ENERGY)
    premises = pd.read_csv(PREP_PREMISES_ENERGY)
    keys = positives.premise_id.to_numpy(dtype=str)
    values = positives.claim_id.values
    gt_mapping = dict(zip(keys, values))
    list = []
    for k, v in mapping.items():
        list.append((v, gt_mapping[k]))
    negatives = pd.DataFrame(list, columns=columns)

    negatives_claims = negatives.merge(claims, on="claim_id")
    negatives_claims = negatives_claims.apply(pd.to_numeric, errors="ignore")
    negatives_complete = negatives_claims.merge(premises, on="premise_id")
    negatives_complete["label"] = "unsimilar"

    positives_claims = positives.merge(claims, on="claim_id")
    positives_complete = positives_claims.merge(premises, on="premise_id")
    positives_complete["label"] = "similar"

    positives_complete = positives_complete[["premise_id", "claim_id", "claim_text", "premise_text", "label"]]
    negative_complete = negatives_complete[["premise_id", "claim_id", "claim_text", "premise_text", "label"]]
    combined = pd.concat([positives_complete, negative_complete])
    combined = combined.drop_duplicates(subset=["claim_id", "premise_id"], keep="first")
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


def generate_mapping():
    representations = torch.load(PREMISES_TEST_FEATURES_ENERGY)
    cos = nn.CosineSimilarity(dim=1)
    # device = "cuda:0"
    dicts = {}
    for k1, v1 in tqdm.tqdm(representations.items(), unit='premises', unit_scale=True):
        # v1.to(device=device)
        best_sim = 0
        best_key = None
        for k2, v2 in representations.items():
            # v2.to(device=device)
            sim = cos(v1.unsqueeze(dim=1), v2.unsqueeze(dim=1)).mean()
            if sim < 1:
                if sim > best_sim:
                    best_sim = sim
                    best_key = k2
        dicts[k1] = best_key
    torch.save(dicts, "mapping.pt")


def generate_mapping1():
    representations = torch.load(PREMISES_TEST_FEATURES_ENERGY)
    #device = "cuda:0"
    #for key, value in representations.items():
    #    representations[key] = representations[key].to(device)
    outer_dict = {}
    for k1, v1 in tqdm.tqdm(representations.items(), unit='premises', unit_scale=True):
        inner_dict = {}
        for k2, v2 in representations.items():
            inner_dict[k2] = torch.cosine_similarity(v1.view(1, -1), v2.view(1, -1)).item()
        outer_dict[k1] = inner_dict
    torch.save(outer_dict, "mapping1.pt")


if __name__ == '__main__':
    main()
