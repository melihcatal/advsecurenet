from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from advsecurenet.dataloader.data_loader_factory import DataLoaderFactory
from advsecurenet.datasets.dataset_factory import DatasetFactory
from advsecurenet.distributed.ddp_base_task import DDPBaseTask
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types.configs import TrainConfig
from advsecurenet.shared.types.configs.device_config import DeviceConfig
from advsecurenet.shared.types.configs.model_config import CreateModelConfig
from advsecurenet.shared.types.configs.preprocess_config import (
    PreprocessConfig, PreprocessStep)
from advsecurenet.shared.types.configs.train_config import TrainConfig
from advsecurenet.trainer.ddp_trainer import DDPTrainer
from advsecurenet.trainer.trainer import Trainer


@pytest.fixture
def processor(request):
    device_arg = request.config.getoption("--device")
    return torch.device(device_arg if device_arg else "cpu")


@pytest.fixture
def train_config(processor):

    model = ModelFactory.create_model(
        model_name='resnet18', num_classes=10, pretrained=False)
    dataset = DatasetFactory.create_dataset(dataset_type="cifar10")
    test_data = dataset.load_dataset(train=False)
    dataloader = DataLoaderFactory.create_dataloader(
        dataset=test_data, batch_size=32)

    # Define the training config
    config = TrainConfig(
        model=model,
        train_loader=dataloader,
        epochs=2,
        processor=processor,
        optimizer='adam'
    )
    return config


@pytest.fixture
@patch('advsecurenet.trainer.ddp_trainer.DDPBaseTask._setup_device')
@patch('advsecurenet.trainer.ddp_trainer.DDPBaseTask._setup_model')
@patch('advsecurenet.trainer.trainer.Trainer._get_optimizer')
def ddp_trainer(mock_optimizer, mock_setup_model, mock_setup_device, train_config):
    rank = 0
    world_size = 2
    return DDPTrainer(config=train_config, rank=rank, world_size=world_size)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_init(ddp_trainer, train_config):
    assert ddp_trainer._rank == 0
    assert ddp_trainer._world_size == 2
    assert ddp_trainer._config == train_config


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_load_model_state_dict(ddp_trainer):
    state_dict = {'key': 'value'}
    ddp_trainer._model = MagicMock()
    ddp_trainer._model.module = MagicMock()
    ddp_trainer._load_model_state_dict(state_dict)
    ddp_trainer._model.module.load_state_dict.assert_called_once_with(
        state_dict)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_model_state_dict(ddp_trainer):
    state_dict = {'key': 'value'}
    ddp_trainer._model = MagicMock()
    ddp_trainer._model.module.state_dict.return_value = state_dict
    result = ddp_trainer._get_model_state_dict()
    assert result == state_dict


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_assign_device_to_optimizer_state(ddp_trainer):
    # Mock the optimizer and its state
    mock_optimizer = MagicMock()
    ddp_trainer._optimizer = mock_optimizer
    mock_tensor = MagicMock(spec=torch.Tensor)
    mock_optimizer.state = {
        0: {'param': mock_tensor},
        1: {'param': "not_tensor"},
    }

    # Mock the rank
    rank = 0
    ddp_trainer._rank = rank
    # Mock the cuda method of torch.Tensor
    with patch.object(mock_tensor, 'cuda', return_value='cuda_tensor') as mock_cuda:
        # Call the method
        ddp_trainer._assign_device_to_optimizer_state()

        # Check if cuda method was called with the correct rank
        mock_cuda.assert_called_once_with(rank)

        # Check if the state was updated correctly
        assert mock_optimizer.state[0]['param'] == 'cuda_tensor'

        assert mock_optimizer.state[1]['param'] == "not_tensor"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_save_checkpoint_prefix(ddp_trainer):
    ddp_trainer._config.save_checkpoint_name = "checkpoint"
    assert ddp_trainer._get_save_checkpoint_prefix() == "checkpoint"

    ddp_trainer._config.save_checkpoint_name = None
    ddp_trainer._config.model.model_name = "model"
    ddp_trainer._config.train_loader.dataset.__class__.__name__ = "dataset"
    assert ddp_trainer._get_save_checkpoint_prefix() == "model_dataset_checkpoint"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_should_save_checkpoint(ddp_trainer):
    epoch = 10
    ddp_trainer._config.checkpoint_interval = 5
    ddp_trainer._config.save_checkpoint = True
    assert ddp_trainer._should_save_checkpoint(epoch) == True

    ddp_trainer._config.checkpoint_interval = 3
    assert ddp_trainer._should_save_checkpoint(epoch) == False


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_should_save_final_model(ddp_trainer):
    ddp_trainer._config.save_final_model = True
    assert ddp_trainer._should_save_final_model() == True

    ddp_trainer._config.save_final_model = False
    assert ddp_trainer._should_save_final_model() == False


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('tqdm.auto.tqdm', wraps=tqdm)
def test_run_epoch(mock_tqdm, ddp_trainer, processor):
    ddp_trainer._config.train_loader = MagicMock(spec=DataLoader)
    ddp_trainer._config.train_loader.sampler = MagicMock(
        spec=DistributedSampler)
    ddp_trainer._run_batch = MagicMock(return_value=1.0)
    ddp_trainer._log_loss = MagicMock()
    ddp_trainer._config.train_loader.__len__.return_value = 1

    ddp_trainer._device = processor
    epoch = 1
    ddp_trainer._run_epoch(epoch)

    ddp_trainer._config.train_loader.sampler.set_epoch.assert_called_once_with(
        epoch)
