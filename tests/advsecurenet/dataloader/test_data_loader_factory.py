from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.sampler import RandomSampler

from advsecurenet.dataloader.data_loader_factory import DataLoaderFactory
from advsecurenet.shared.types.configs.dataloader_config import \
    DataLoaderConfig


class MockDataset(TorchDataset):
    def __len__(self):
        return 10

    def __getitem__(self, item):
        return torch.zeros(1)


@pytest.fixture
def mock_dataset():
    return MockDataset()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_dataloader_with_config(mock_dataset):
    config = DataLoaderConfig(
        dataset=mock_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        sampler=None
    )
    dataloader = DataLoaderFactory.create_dataloader(config)
    assert isinstance(dataloader, TorchDataLoader)
    assert dataloader.dataset == mock_dataset
    assert dataloader.batch_size == 32
    assert dataloader.num_workers == 4
    assert dataloader.drop_last is True
    assert dataloader.pin_memory is True
    assert isinstance(dataloader.sampler, RandomSampler)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_dataloader_without_config(mock_dataset):
    dataloader = DataLoaderFactory.create_dataloader(
        dataset=mock_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        sampler=None
    )
    assert isinstance(dataloader, TorchDataLoader)
    assert dataloader.dataset == mock_dataset
    assert dataloader.batch_size == 32
    assert dataloader.num_workers == 4
    assert dataloader.drop_last is True
    assert dataloader.pin_memory is True
    assert isinstance(dataloader.sampler, RandomSampler)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_dataloader_invalid_dataset_type():
    with pytest.raises(ValueError):
        DataLoaderFactory.create_dataloader(dataset="invalid_dataset_type")


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_dataloader_with_sampler_and_shuffle(mock_dataset):
    config = DataLoaderConfig(
        dataset=mock_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        sampler=Mock()
    )
    dataloader = DataLoaderFactory.create_dataloader(config)
    assert isinstance(dataloader, TorchDataLoader)
    assert dataloader.dataset == mock_dataset
    assert dataloader.batch_size == 32
    assert dataloader.num_workers == 4
    # shuffle should be False when sampler is provided
    assert config.shuffle is False
    assert dataloader.drop_last is True
    assert dataloader.pin_memory is True
    assert isinstance(dataloader.sampler, Mock)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_dataloader_with_sampler_and_no_shuffle(mock_dataset):
    config = DataLoaderConfig(
        dataset=mock_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        sampler=Mock()
    )
    dataloader = DataLoaderFactory.create_dataloader(config)
    assert isinstance(dataloader, TorchDataLoader)
    assert dataloader.dataset == mock_dataset
    assert dataloader.batch_size == 32
    assert dataloader.num_workers == 4
    # shuffle should be False when sampler is provided
    assert config.shuffle is False
    assert dataloader.drop_last is True
    assert dataloader.pin_memory is True
    assert isinstance(dataloader.sampler, Mock)
