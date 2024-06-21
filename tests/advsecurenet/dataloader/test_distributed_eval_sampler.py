# test_distributed_eval_sampler.py

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from advsecurenet.dataloader.distributed_eval_sampler import \
    DistributedEvalSampler


class MockDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.tensor(idx)


@pytest.fixture
def mock_dataset():
    return MockDataset(size=100)


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('torch.distributed.is_available', return_value=True)
@patch('torch.distributed.get_world_size', return_value=4)
@patch('torch.distributed.get_rank', return_value=1)
def test_distributed_eval_sampler_initialization(mock_is_available, mock_get_world_size, mock_get_rank, mock_dataset):
    sampler = DistributedEvalSampler(mock_dataset, shuffle=True)
    assert sampler.num_replicas == 4
    assert sampler.rank == 1
    assert sampler.shuffle is True
    assert sampler.total_size == len(mock_dataset)
    assert sampler.num_samples == len(mock_dataset) // 4


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('torch.distributed.is_available', return_value=True)
@patch('torch.distributed.get_world_size', return_value=4)
@patch('torch.distributed.get_rank', return_value=1)
def test_distributed_eval_sampler_length(mock_is_available, mock_get_world_size, mock_get_rank, mock_dataset):
    sampler = DistributedEvalSampler(mock_dataset)
    assert len(sampler) == len(mock_dataset) // 4


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('torch.distributed.is_available', return_value=True)
@patch('torch.distributed.get_world_size', return_value=4)
@patch('torch.distributed.get_rank', return_value=1)
def test_distributed_eval_sampler_iteration(mock_is_available, mock_get_world_size, mock_get_rank, mock_dataset):
    sampler = DistributedEvalSampler(mock_dataset, shuffle=False)
    indices = list(iter(sampler))
    expected_indices = list(range(1, len(mock_dataset), 4))
    assert indices == expected_indices


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('torch.distributed.is_available', return_value=True)
@patch('torch.distributed.get_world_size', return_value=4)
@patch('torch.distributed.get_rank', return_value=1)
def test_distributed_eval_sampler_shuffle(mock_is_available, mock_get_world_size, mock_get_rank, mock_dataset):
    sampler = DistributedEvalSampler(mock_dataset, shuffle=True, seed=42)
    sampler.set_epoch(0)
    indices_epoch_0 = list(iter(sampler))

    sampler.set_epoch(1)
    indices_epoch_1 = list(iter(sampler))

    # Check that the indices are different across epochs when shuffle is True
    assert indices_epoch_0 != indices_epoch_1


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('torch.distributed.is_available', return_value=True)
@patch('torch.distributed.get_world_size', return_value=4)
@patch('torch.distributed.get_rank', return_value=1)
def test_distributed_eval_sampler_set_epoch(mock_is_available, mock_get_world_size, mock_get_rank, mock_dataset):
    sampler = DistributedEvalSampler(mock_dataset, shuffle=True, seed=42)
    sampler.set_epoch(5)
    assert sampler.epoch == 5
