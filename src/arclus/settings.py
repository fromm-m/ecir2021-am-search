"""
Global settings.
"""
import pathlib

# Preprocessing Training-Data
DATA_ROOT = pathlib.Path('/nfs/data3/fromm/argument_clustering/data')
# TODO: change to real output path
DATA_EXPERIMENTS = pathlib.Path('/nfs/data3/obermeier/arclus/experiments')
PREP_ASSIGNMENTS = DATA_ROOT / 'preprocessed' / 'assignments.csv'
PREP_CLAIMS = DATA_ROOT / 'preprocessed' / 'claims.csv'
PREP_PREMISES = DATA_ROOT / 'preprocessed' / 'premises.csv'

# Training preparations
TRAIN_SIZE = 0.7
TEST_SIZE = 0.2
VALID_SIZE = 1 - TRAIN_SIZE - TEST_SIZE

TRAIN_PATH = DATA_ROOT / 'training' / 'train.csv'
TEST_PATH = DATA_ROOT / 'training' / 'test.csv'
VALID_PATH = DATA_ROOT / 'training' / 'valid.csv'

# Preprocessing Test-Data
PREP_ASSIGNMENTS_TEST = DATA_ROOT / 'preprocessed' / 'old_filtered.csv'
PREP_CLAIMS_TEST = DATA_ROOT / 'preprocessed' / 'claim_test.csv'
PREP_PREMISES_TEST = DATA_ROOT / 'preprocessed' / 'premises_test.csv'

# Feature Generation
CLAIMS_LENGTH = 19
PREMISES_LENGTH = 220
OUTPUT_FEATURES_FOLDER = DATA_ROOT / 'features'
OUTPUT_FEATURES_POSITIVES = DATA_ROOT / 'features' / 'embeddings_positives.npy'
OUTPUT_FEATURES_NEGATIVE = DATA_ROOT / 'features' / 'embeddings_negatives.npy'
CLAIMS_TEST_FEATURES = DATA_ROOT / 'features' / 'claims_test_embeddings.pt'
PREMISES_TEST_FEATURES = DATA_ROOT / 'features' / 'premises_test_embeddings.pt'
# Negative Sampling
NEGATIVE_SAMPLES = DATA_ROOT / 'preprocessed' / 'negatives.csv'
# one negative per positive
NUM_NEG_PER_POS = 1

# Similarities
PREP_TEST_SIMILARITIES = DATA_ROOT / 'similarities' / 'test_similarities.pt'
