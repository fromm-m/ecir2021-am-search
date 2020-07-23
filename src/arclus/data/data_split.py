import logging
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler

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


class DataSplit:
    """ General class for splitting data into train, test and validation set. """

    def __init__(
        self,
        dataset: Dataset,
        train_val_fraction: int = TRAIN_SIZE,
        val_fraction: int = VAL_SIZE,
        shuffle=True
    ):
        """
        Initialize samplers.
        :param dataset:
        :param train_val_fraction: fraction of the train & validation set
        :param val_fraction:
        :param shuffle: Whether to shuffle or not
        """
        self.dataset = dataset
        dataset_size = len(dataset)
        self.indices = list(range(dataset_size))
        # index of where to split train and test
        test_split = int(np.floor(train_val_fraction * dataset_size))

        if shuffle:
            np.random.shuffle(self.indices)

        # train set is from 0-test_split and test set from test_split- dataset size
        train_indices, self.test_indices = self.indices[:test_split], self.indices[test_split:]
        # validation split is the index of where to split train and validation
        val_split = int(np.floor((1 - val_fraction) * len(train_indices)))
        # train set is from 0-validation_split and validation set from validation_split - train set size
        self.train_indices, self.val_indices = train_indices[:val_split], train_indices[val_split:]

        self.train_sampler = SubsetRandomSampler(indices=self.train_indices)
        self.val_sampler = SubsetRandomSampler(indices=self.val_indices)
        self.test_sampler = SubsetRandomSampler(indices=self.test_indices)

    def get_train_split_point(self):
        return len(self.train_sampler) + len(self.val_sampler)

    def get_validation_split_point(self):
        return len(self.train_sampler)

    def get_train_val_test_split(
        self,
        batch_size: int,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Split the data into train, validation and test set.
        :param: batch_size
        :param: num_workers
        :return one torch Dataloader for train set, validation set and test set (in that order) each.
        """
        logging.debug('Initialize dataloaders for train, validation and test')

        loaders = []
        for sampler in [self.train_sampler, self.val_sampler, self.test_sampler]:
            loaders.append(torch.utils.data.DataLoader(dataset=self.dataset, batch_size=batch_size,
                                                       sampler=sampler,
                                                       shuffle=False, num_workers=num_workers))
        train_loader, val_loader, test_loader = tuple(loaders)
        return train_loader, val_loader, test_loader
