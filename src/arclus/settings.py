"""
Global settings.
"""
import pathlib

TRAIN_SIZE = 0.8
VAL_SIZE = 0.1

DATA_ROOT = pathlib.Path('/nfs/data3/fromm/argument_clustering/data')
PREP_ASSIGNMENTS = DATA_ROOT / 'preprocessed' / 'assignments.csv'
PREP_CLAIMS = DATA_ROOT / 'preprocessed' / 'claims.csv'
PREP_PREMISES = DATA_ROOT / 'preprocessed' / 'premises.csv'
OUTPUT_FEATURES = DATA_ROOT / 'features' / 'embeddings.npy'
OUTPUT_FEATURES_NEGATIVE = DATA_ROOT / 'features' / 'embeddings_negatives.npy'
CLAIMS_LENGTH = 19
PREMISES_LENGTH = 220
