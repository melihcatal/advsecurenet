from advsecurenet.datasets.dataset_factory import DatasetFactory, DatasetType
from advsecurenet.datasets.base_dataset import BaseDataset
import pytest


class TestDatasetFactory:
    def test_load_dataset(self):
        # Test loading MNIST dataset
        dataset = DatasetFactory.load_dataset(DatasetType.MNIST)
        assert isinstance(dataset, BaseDataset)
        assert dataset.num_classes == 10
        assert dataset.input_size == (28, 28)

        # Test loading CIFAR10 dataset
        dataset = DatasetFactory.load_dataset(DatasetType.CIFAR10)
        assert isinstance(dataset, BaseDataset)
        assert dataset.num_classes == 10
        assert dataset.input_size == (32, 32)
