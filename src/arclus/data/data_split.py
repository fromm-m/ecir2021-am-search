from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from arclus.settings import OUTPUT_FEATURES, OUTPUT_FEATURES_NEGATIVE, TRAIN_SIZE, VAL_SIZE


class PrecomputedPairwiseFeatures(Dataset):
    def __init__(self):
        """
        Load positive and negative output features and generate labels (1 for positive, 0 for negative)
        """
        x_data_pos = np.load(OUTPUT_FEATURES)
        x_data_neg = np.load(OUTPUT_FEATURES_NEGATIVE)
        y_data_pos = torch.ones(len(x_data_pos), dtype=torch.long)
        y_data_neg = torch.zeros(len(x_data_neg), dtype=torch.long)
        self.X_data = np.concatenate((x_data_pos, x_data_neg))
        self.y_data = np.concatenate((y_data_pos, y_data_neg))

    def info(self) -> dict:
        n_samples, dim = self.X_data.shape
        n_pos = (self.y_data > 0).sum()
        return dict(
            samples_total=n_samples,
            samples_neg=n_samples - n_pos,
            samples_pos=n_pos,
            feature_dim=dim,
        )

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def split(
    dataset: Dataset,
    train_test_ratio: int = TRAIN_SIZE,
    train_validation_ratio: int = VAL_SIZE,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Split the dataset into train-validation-test."""
    n_samples = len(dataset)
    train_size = int(train_test_ratio * n_samples)
    test_size = n_samples - train_size
    train_size = int(train_validation_ratio * n_samples)
    validation_size = n_samples - train_size - test_size
    return random_split(dataset=dataset, lengths=[train_size, validation_size, test_size])


def get_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Get a data loader."""
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
