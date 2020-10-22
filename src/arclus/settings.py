"""
Global settings.
"""
import pathlib

# Preprocessing Training-Data
DATA_ROOT = pathlib.Path('/nfs/data3/fromm/argument_clustering/data')
# TODO: change to real output path
DATA_EXPERIMENTS = pathlib.Path('/nfs/data3/obermeier/arclus/experiments')
PREP_ASSIGNMENTS_ENERGY = DATA_ROOT / 'preprocessed' / 'energy_filtered_assignments.csv'
PREP_CLAIMS_ENERGY = DATA_ROOT / 'preprocessed' / 'energy_filtered_claims.csv'
PREP_PREMISES_ENERGY = DATA_ROOT / 'preprocessed' / 'energy_filtered_premises.csv'

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

TRAIN_PATH_ENERGY = DATA_ROOT / 'training' / 'train_energy_nn.csv'
TEST_PATH_ENERGY = DATA_ROOT / 'training' / 'test_energy_nn.csv'
VALID_PATH_ENERGY = DATA_ROOT / 'training' / 'valid_energy_nn.csv'

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
PREMISES_FEATURES = DATA_ROOT / 'features' / 'premises_embeddings.pt'
PREMISES_FEATURES_ENERGY = DATA_ROOT / 'features' / 'premises_embeddings_energy.pt'
# Negative Sampling
NEGATIVE_SAMPLES = DATA_ROOT / 'preprocessed' / 'negatives.csv'
# one negative per positive
NUM_NEG_PER_POS = 1

# Similarities
PREP_TEST_SIMILARITIES = 'test_similarities.pt'
PREP_TEST_STATES = 'test_states.pt'
PREP_TEST_PRODUCT_SIMILARITIES = 'test_product_similarities.pt'
