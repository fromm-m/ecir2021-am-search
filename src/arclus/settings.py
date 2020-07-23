"""
Global settings.
"""
import pathlib
# Preprocessing Training-Data
DATA_ROOT = pathlib.Path('/nfs/data3/fromm/argument_clustering/data')
PREP_ASSIGNMENTS = DATA_ROOT / 'preprocessed' / 'assignments.csv'
PREP_CLAIMS = DATA_ROOT / 'preprocessed' / 'claims.csv'
PREP_PREMISES = DATA_ROOT / 'preprocessed' / 'premises.csv'

# Preprocessing Test-Data
PREP_ASSIGNMENTS_TEST = DATA_ROOT / 'preprocessed' / 'processedPremises.csv'
PREP_CLAIMS_TEST = DATA_ROOT / 'preprocessed' / 'claim_test.csv'
PREP_PREMISES_TEST = DATA_ROOT / 'preprocessed' / 'premises_test.csv'

# Feature Generation
CLAIMS_LENGTH = 19
PREMISES_LENGTH = 220
OUTPUT_FEATURES = DATA_ROOT / 'features' / 'embeddings.npy'
CLAIMS_TEST_FEATURES = DATA_ROOT / 'features' / 'claims_test_embeddings.npy'
PREMISES_TEST_FEATURES = DATA_ROOT / 'features' / 'premises_test_embeddings.npy'


# Negative Sampling
NEGATIVE_SAMPLES = DATA_ROOT / 'preprocessed' / 'negatives.csv'
# one negative per positive
NUM_NEG_PER_POS = 1
