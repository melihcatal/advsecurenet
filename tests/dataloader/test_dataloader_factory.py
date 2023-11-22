import pytest
import torch
from torch.utils.data import Dataset, DataLoader
from advsecurenet.datasets import BaseDataset
from advsecurenet.dataloader import DataLoaderFactory


class MockDataset(BaseDataset, Dataset):
    def __init__(self):
        super().__init__()
        self.name = "mock_dataset"
        self.input_size = (32, 32)
        self.num_classes = 10
        self.num_input_channels = 3

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return torch.randn(self.num_input_channels, *self.input_size), idx % self.num_classes

    def load_dataset(self, root, train, download, **kwargs):
        return self


def test_create_dataloader_defaults():
    dataset_instance = MockDataset()
    dataloader = DataLoaderFactory.create_dataloader(dataset_instance)

    assert isinstance(dataloader, DataLoader)
    assert len(dataloader.dataset) == 100
    assert dataloader.batch_size == 32
    assert isinstance(dataloader.sampler, torch.utils.data.SequentialSampler)
    assert dataloader.num_workers == 4


def test_create_dataloader_with_shuffle():
    dataset_instance = MockDataset()
    dataloader = DataLoaderFactory.create_dataloader(
        dataset_instance, shuffle=True)

    assert isinstance(dataloader.sampler, torch.utils.data.RandomSampler)


def test_create_dataloader_with_custom_batch_size():
    dataset_instance = MockDataset()
    dataloader = DataLoaderFactory.create_dataloader(
        dataset_instance, batch_size=16)

    assert dataloader.batch_size == 16


def test_create_dataloader_with_invalid_dataset_type():
    with pytest.raises(ValueError) as e:
        dataloader = DataLoaderFactory.create_dataloader("invalid_dataset")

    assert str(e.value) == "Invalid dataset type provided. Expected TorchDataset."
