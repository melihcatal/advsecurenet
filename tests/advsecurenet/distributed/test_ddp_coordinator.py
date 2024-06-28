import os
from unittest.mock import MagicMock, patch

import pytest

from advsecurenet.distributed.ddp_coordinator import DDPCoordinator


@pytest.fixture
def mock_train_func():
    return MagicMock()


@pytest.fixture
@patch('advsecurenet.distributed.ddp_coordinator.mp.set_start_method')
def ddp_training_coordinator(mock_set_start_method, mock_train_func):
    return DDPCoordinator(
        ddp_func=mock_train_func,
        world_size=2
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.distributed.ddp_coordinator.find_free_port', return_value=12345)
@patch('advsecurenet.distributed.ddp_coordinator.mp.set_start_method')
def test_initialization(mock_set_start_method, mock_find_free_port, mock_train_func):
    coordinator = DDPCoordinator(
        ddp_func=mock_train_func,
        world_size=2,
    )
    assert coordinator.port == 12345
    assert coordinator.backend == 'nccl'
    assert os.environ['MASTER_ADDR'] == 'localhost'
    assert os.environ['MASTER_PORT'] == '12345'


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.distributed.ddp_coordinator.init_process_group')
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
@patch('advsecurenet.distributed.ddp_coordinator.init_process_group')
@patch('advsecurenet.distributed.ddp_coordinator.destroy_process_group')
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
@patch('advsecurenet.distributed.ddp_coordinator.mp.spawn')
def test_run(mock_spawn, ddp_training_coordinator):
    ddp_training_coordinator.run()
    mock_spawn.assert_called_once_with(
        ddp_training_coordinator.run_process,
        nprocs=ddp_training_coordinator.world_size,
        join=True
    )
