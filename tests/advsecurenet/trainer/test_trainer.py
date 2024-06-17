import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types.configs.train_config import TrainConfig
from advsecurenet.trainer.trainer import Trainer


@pytest.fixture
def device(request):
    device_arg = request.config.getoption("--device")
    return torch.device(device_arg if device_arg else "cpu")


@pytest.fixture
def train_config(device):
    model = ModelFactory.create_model(
        model_name="CustomCifar10Model", pretrained=False)

    dataset = TensorDataset(torch.randn(100, 3, 32, 32),
                            torch.randint(0, 10, (100,)))
    train_loader = DataLoader(dataset, batch_size=10)
    config = TrainConfig(
        model=model,
        train_loader=train_loader,
        processor=device,
        learning_rate=0.001,
        epochs=1,
        checkpoint_interval=1,
        save_checkpoint=True,
        save_final_model=True,
        save_checkpoint_path="./checkpoints",
        save_model_path="./models",
        use_ddp=False,
    )
    return config


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_trainer_initialization(train_config, device):
    trainer = Trainer(train_config)
    assert trainer._config == train_config
    assert trainer._device == device
    assert trainer._model == train_config.model
    assert trainer._optimizer is not None
    assert trainer._loss_fn is not None
    assert trainer._scheduler is None


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_setup_device(train_config, device):
    trainer = Trainer(train_config)
    trainer_device = trainer._setup_device()
    assert trainer_device == device


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_setup_model(train_config, device):
    trainer = Trainer(train_config)
    model = trainer._setup_model()
    assert model == train_config.model
    assert next(model.parameters()).device.type == device.type


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_setup_optimizer(train_config):
    trainer = Trainer(train_config)
    optimizer = trainer._setup_optimizer()
    assert isinstance(optimizer, optim.Adam)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_setup_scheduler(train_config):
    train_config.scheduler = "LINEAR_LR"
    trainer = Trainer(train_config)
    scheduler = trainer._setup_scheduler()
    assert isinstance(scheduler, torch.optim.lr_scheduler.LinearLR)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_loss_function(train_config):
    trainer = Trainer(train_config)
    loss_fn = trainer._get_loss_function(train_config.criterion)
    assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("torch.save")
def test_save_checkpoint(mock_save, train_config):
    trainer = Trainer(train_config)
    trainer._save_checkpoint(epoch=1, optimizer=trainer._optimizer)
    assert mock_save.called


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_should_save_checkpoint(train_config):
    trainer = Trainer(train_config)
    assert trainer._should_save_checkpoint(epoch=1) is True


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("torch.save")
def test_save_final_model(mock_save, train_config):
    trainer = Trainer(train_config)
    trainer._save_final_model()
    assert mock_save.called


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
def test_run_batch(train_config, device):
    trainer = Trainer(train_config)
    source = torch.randn((10, 3, 32, 32), dtype=torch.float32, device=device)
    targets = torch.randint(0, 10, (10,), device=device)
    loss = trainer._run_batch(source, targets)
    assert isinstance(loss, float)


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
def test_run_epoch(train_config):
    trainer = Trainer(train_config)
    trainer._run_epoch(epoch=1)
    assert True  # Check if the function runs without errors


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_pre_training(train_config):
    trainer = Trainer(train_config)
    trainer._pre_training()
    assert trainer._model.training is True


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.trainer.trainer.Trainer._save_final_model", return_value=None)
def test_post_training(mock_save_final_model, train_config):
    train_config.save_final_model = True
    trainer = Trainer(train_config)
    trainer._post_training()
    assert mock_save_final_model.called


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_log_loss(train_config, tmp_path):
    trainer = Trainer(train_config)
    trainer._log_loss(epoch=1, loss=0.61, dir=tmp_path, filename="loss.log")
    assert os.path.exists(tmp_path / "loss.log")
    with open(tmp_path / "loss.log", "r") as f:
        assert f.readline() == "epoch,loss\n"
        assert f.readline() == "1,0.61\n"
