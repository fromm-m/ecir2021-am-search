import pandas as pd
import numpy as np

from arclus.settings import PREP_PREMISES, PREP_CLAIMS, PREP_ASSIGNMENTS, NEGATIVE_SAMPLES, \
    TEST_SIZE, VALID_SIZE, TRAIN_PATH, TEST_PATH, VALID_PATH

premises_df = pd.read_csv(PREP_PREMISES, delimiter=",")
claims_df = pd.read_csv(PREP_CLAIMS, delimiter=",")
pos_df = pd.read_csv(PREP_ASSIGNMENTS, delimiter=",")
neg_df = pd.read_csv(NEGATIVE_SAMPLES, delimiter=",")

positives = pd.merge(pos_df, premises_df, how='inner', on="premise_id")
positives = pd.merge(positives, claims_df, how='inner', on="claim_id")
positives["id"] = positives["claim_id"].astype(str) + "_" + positives["premise_id"].astype(str)
positives["annotation"] = "similar"
positives = positives.drop(
    columns=['Unnamed: 0_x', 'Unnamed: 0_y', "Unnamed: 0", "stance", "source_x", "source_y", "premise_id", "claim_id"])

negatives = pd.merge(neg_df, premises_df, how='inner', on="premise_id")
negatives = pd.merge(negatives, claims_df, how='inner', on="claim_id")
negatives["id"] = negatives["claim_id"].astype(str) + "_" + negatives["premise_id"].astype(str)
negatives["annotation"] = "unsimilar"
negatives = negatives.drop(
    columns=['Unnamed: 0_x', 'Unnamed: 0_y', "Unnamed: 0", "stance", "source_x", "source_y", "premise_id", "claim_id"])

data = pd.concat([positives, negatives])
data = data.sample(frac=1).reset_index(drop=True)

mask1 = np.random.rand(len(data)) < 1 - TEST_SIZE
df_train_valid_set = data[mask1]
df_test_set = data[~mask1]

mask2 = np.random.rand(len(df_train_valid_set)) < 1 - VALID_SIZE
df_train_set = df_train_valid_set[mask2]
df_valid_set = df_train_valid_set[~mask2]

print("train", len(df_train_set))
print("test", len(df_test_set))
print("valid", len(df_valid_set))

df_train_set.to_csv(TRAIN_PATH)
df_test_set.to_csv(TEST_PATH)
df_valid_set.to_csv(VALID_PATH)
