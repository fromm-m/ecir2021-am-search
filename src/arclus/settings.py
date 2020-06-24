"""
Global settings.
"""
import pathlib

DATA_ROOT = pathlib.Path('/nfs/data3/fromm/argument_clustering/data')
PREP_ASSIGNMENTS = DATA_ROOT / 'preprocessed' / 'assignments.csv'
PREP_CLAIMS = DATA_ROOT / 'preprocessed' / 'claims.csv'
PREP_PREMISES = DATA_ROOT / 'preprocessed' / 'premises.csv'
OUTPUT_FEATURES = DATA_ROOT / 'features' / 'embeddings.npy'
CLAIMS_LENGTH = 19
PREMISES_LENGTH = 220
