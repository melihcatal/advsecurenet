import os
from unittest.mock import MagicMock, patch

import pytest
import torch.multiprocessing as mp

from advsecurenet.trainer.ddp_training_coordinator import (
    DDPTrainingCoordinator, destroy_process_group, init_process_group)
from advsecurenet.utils.network import find_free_port


@pytest.fixture
def mock_train_func():
    return MagicMock()


@pytest.fixture
def ddp_training_coordinator(mock_train_func):
    return DDPTrainingCoordinator(
        train_func=mock_train_func,
        world_size=2
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.trainer.ddp_training_coordinator.find_free_port', return_value=12345)
def test_initialization(mock_find_free_port, mock_train_func):
    coordinator = DDPTrainingCoordinator(
        train_func=mock_train_func,
        world_size=2,
    )
    assert coordinator.port == 12345
    assert coordinator.backend == 'nccl'
    assert os.environ['MASTER_ADDR'] == 'localhost'
    assert os.environ['MASTER_PORT'] == '12345'


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.trainer.ddp_training_coordinator.init_process_group')
def test_ddp_setup(mock_init_process_group, ddp_training_coordinator):
    rank = 0
    ddp_training_coordinator.ddp_setup(rank)
    mock_init_process_group.assert_called_once_with(
        backend='nccl',
        rank=rank,
        world_size=ddp_training_coordinator.world_size
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.trainer.ddp_training_coordinator.init_process_group')
@patch('advsecurenet.trainer.ddp_training_coordinator.destroy_process_group')
def test_run_process(mock_destroy_process_group, mock_init_process_group, ddp_training_coordinator, mock_train_func):
    rank = 0
    ddp_training_coordinator.run_process(rank)
    mock_init_process_group.assert_called_once_with(
        backend='nccl',
        rank=rank,
        world_size=ddp_training_coordinator.world_size
    )
    mock_train_func.assert_called_once_with(
        rank, ddp_training_coordinator.world_size, *ddp_training_coordinator.args, **ddp_training_coordinator.kwargs
    )
    mock_destroy_process_group.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('torch.multiprocessing.Process')
def test_run(mock_process, ddp_training_coordinator):
    ddp_training_coordinator.run()
    assert mock_process.call_count == ddp_training_coordinator.world_size
    for i in range(ddp_training_coordinator.world_size):
        mock_process.assert_any_call(
            target=ddp_training_coordinator.run_process,
            args=(i,)
        )
    assert all(p.start.called for p in mock_process.mock_calls)
    assert all(p.join.called for p in mock_process.mock_calls)
