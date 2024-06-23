import logging
import os
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types.configs.train_config import TrainConfig
from advsecurenet.trainer.trainer import Trainer

logger = logging.getLogger("advsecurenet.trainer.trainer")


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
def test_setup_optimizer_with_kwargs(train_config):
    train_config.optimizer = "adam"
    train_config.learning_rate = 0.1
    train_config.optimizer_kwargs = {"betas": (0.9, 0.999)}
    trainer = Trainer(train_config)
    optimizer = trainer._setup_optimizer()

    assert isinstance(optimizer, optim.Adam)
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["lr"] == 0.1


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_setup_scheduler(train_config):
    train_config.scheduler = "LINEAR_LR"
    trainer = Trainer(train_config)
    scheduler = trainer._setup_scheduler()
    assert isinstance(scheduler, torch.optim.lr_scheduler.LinearLR)


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


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.trainer.trainer.Trainer.__init__', return_value=None)
def test_get_optimizer_str_optim_missing_model(mock_init, train_config):
    optimizer = "adam"
    model = None
    trainer = Trainer(train_config)
    with pytest.raises(ValueError) as excinfo:
        trainer._get_optimizer(
            optimizer=optimizer,
            model=model
        )
    assert excinfo.value.args[0] == "Model must be provided if optimizer is a string."


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.trainer.trainer.Trainer.__init__', return_value=None)
def test_get_optimizer_optimizer_optim(mock_init, train_config):
    from torch.optim import Adam

    optimizer = Adam(train_config.model.parameters())
    model = train_config.model

    trainer = Trainer(train_config)

    returned_optimizer = trainer._get_optimizer(
        optimizer=optimizer,
        model=model
    )
    assert returned_optimizer == optimizer


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.trainer.trainer.Trainer.__init__', return_value=None)
def test_get_optimizer_default_optimizer(mock_init, train_config):
    from torch.optim import Adam

    expected_default_optimizer = Adam(train_config.model.parameters())
    expected_default_lr = 0.001

    optimizer = None
    model = train_config.model

    trainer = Trainer(train_config)

    returned_optimizer = trainer._get_optimizer(
        optimizer=optimizer,
        model=model
    )

    assert isinstance(returned_optimizer, Adam)
    assert returned_optimizer.defaults["lr"] == expected_default_lr


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.trainer.trainer.Trainer.__init__', return_value=None)
def test_get_optimizer_str_optim(mock_init, train_config):
    from torch.optim import Adam

    optimizer = "adam"
    model = train_config.model
    lr = 1

    trainer = Trainer(train_config)

    returned_optimizer = trainer._get_optimizer(
        optimizer=optimizer,
        model=model,
        learning_rate=lr
    )

    assert isinstance(returned_optimizer, Adam)
    assert returned_optimizer.defaults["lr"] == lr


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.trainer.trainer.Trainer.__init__', return_value=None)
def test_get_optimizer_str_optim_invalid(mock_init, train_config):

    optimizer = "NOT_SUPPORTED_OPTIMIZER"
    model = train_config.model
    lr = 1

    trainer = Trainer(train_config)

    with pytest.raises(ValueError) as excinfo:
        trainer._get_optimizer(
            optimizer=optimizer,
            model=model,
            learning_rate=lr
        )

    assert "Unsupported optimizer!" in excinfo.value.args[0]


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.trainer.trainer.Trainer.__init__', return_value=None)
def test_get_optimizer_str_optim_with_kwargs(mock_init, train_config):
    from torch.optim import Adam

    optimizer = "adam"
    model = train_config.model
    lr = 1
    optimizer_kwargs = {"betas": (0.9, 0.999)}

    trainer = Trainer(train_config)

    returned_optimizer = trainer._get_optimizer(
        optimizer=optimizer,
        model=model,
        learning_rate=lr,
        **optimizer_kwargs
    )

    assert isinstance(returned_optimizer, Adam)
    assert returned_optimizer.defaults["lr"] == lr
    assert returned_optimizer.defaults["betas"] == optimizer_kwargs["betas"]


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.trainer.trainer.Trainer.__init__', return_value=None)
def test_get_optimizer_str_optim_with_kwargs(mock_init, train_config):
    from torch.optim import Adam

    optimizer = "adam"
    model = train_config.model
    lr = 1
    optimizer_kwargs = {"betas": (0.9, 0.999)}

    trainer = Trainer(train_config)

    returned_optimizer = trainer._get_optimizer(
        optimizer=optimizer,
        model=model,
        learning_rate=lr,
        **optimizer_kwargs
    )

    assert isinstance(returned_optimizer, Adam)
    assert returned_optimizer.defaults["lr"] == lr
    assert returned_optimizer.defaults["betas"] == optimizer_kwargs["betas"]


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.trainer.trainer.Trainer._setup_device')
@patch('advsecurenet.trainer.trainer.Trainer._setup_model')
@patch('advsecurenet.trainer.trainer.Trainer._setup_optimizer')
@patch('advsecurenet.trainer.trainer.get_loss_function')
@patch('advsecurenet.trainer.trainer.Trainer._load_checkpoint_if_any', return_value=1)
@patch('advsecurenet.trainer.trainer.Trainer._setup_scheduler')
def test_train(mock_setup_scheduler, mock_load_checkpoint_if_any, mock_get_loss_function, mock_setup_optimizer,
               mock_setup_model, mock_setup_device, train_config):
    mock_device = MagicMock()
    mock_setup_device.return_value = mock_device
    mock_model = MagicMock()
    mock_setup_model.return_value = mock_model
    mock_optimizer = MagicMock()
    mock_setup_optimizer.return_value = mock_optimizer
    mock_loss_fn = MagicMock()
    mock_get_loss_function.return_value = mock_loss_fn

    trainer = Trainer(train_config)

    # Mock the methods in the MyTrainer class
    with patch.object(trainer, '_pre_training') as mock_pre_training, \
            patch.object(trainer, '_run_epoch') as mock_run_epoch, \
            patch.object(trainer, '_should_save_checkpoint') as mock_should_save_checkpoint, \
            patch.object(trainer, '_save_checkpoint') as mock_save_checkpoint, \
            patch.object(trainer, '_post_training') as mock_post_training, \
            patch('advsecurenet.trainer.trainer.trange', return_value=range(1, train_config.epochs + 1)):

        # Configure the mocks
        mock_should_save_checkpoint.side_effect = lambda epoch: epoch % 2 == 0

        # Call the train method
        trainer.train()

        # Assertions
        mock_pre_training.assert_called_once()
        assert mock_run_epoch.call_count == train_config.epochs
        for epoch in range(1, train_config.epochs + 1):
            mock_run_epoch.assert_any_call(epoch)
        assert mock_should_save_checkpoint.call_count == train_config.epochs
        for epoch in range(1, train_config.epochs + 1):
            mock_should_save_checkpoint.assert_any_call(epoch)
        assert mock_save_checkpoint.call_count == train_config.epochs // 2
        for epoch in range(2, train_config.epochs + 1, 2):
            mock_save_checkpoint.assert_any_call(epoch, mock_optimizer)
        mock_post_training.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('os.path.isfile', return_value=True)
@patch('torch.load')
@patch.object(Trainer, '_load_model_state_dict')
@patch.object(Trainer, '_assign_device_to_optimizer_state')
def test_load_checkpoint_if_exists(mock_assign_device, mock_load_state_dict, mock_torch_load, mock_isfile, train_config):
    train_config.load_checkpoint = True
    train_config.load_checkpoint_path = '/path/to/checkpoint'
    mock_torch_load.return_value = {
        'model_state_dict': 'mock_model_state_dict',
        'optimizer_state_dict': 'mock_optimizer_state_dict',
        'epoch': 10
    }
    trainer = Trainer(train_config)

    with mock.patch.object(logger, 'info') as mock_logger, \
            mock.patch.object(trainer._optimizer, 'load_state_dict') as mock_load_optimizer_dict:
        start_epoch = trainer._load_checkpoint_if_any()

        assert start_epoch == 11
        mock_isfile.assert_called_with('/path/to/checkpoint')
        mock_torch_load.assert_called_with('/path/to/checkpoint')
        mock_load_state_dict.assert_called_with('mock_model_state_dict')
        mock_load_optimizer_dict.assert_called_with(
            'mock_optimizer_state_dict')
        mock_assign_device.assert_called()
        mock_logger.assert_called_with(
            "Loading checkpoint from %s", '/path/to/checkpoint')


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('os.path.isfile', return_value=False)
def test_load_checkpoint_if_not_exists(mock_isfile, train_config):
    train_config.load_checkpoint = True
    train_config.load_checkpoint_path = '/path/to/checkpoint'

    trainer = Trainer(train_config)
    with mock.patch.object(logger, 'warning') as mock_warning:
        start_epoch = trainer._load_checkpoint_if_any()

        assert start_epoch == 1
        mock_isfile.assert_called_with('/path/to/checkpoint')
        mock_warning.assert_called_with(
            "Checkpoint file not found at %s", '/path/to/checkpoint')


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('os.path.isfile', return_value=True)
@patch('torch.load', side_effect=FileNotFoundError)
def test_load_checkpoint_load_error(mock_torch_load, mock_isfile, train_config):
    train_config.load_checkpoint = True
    train_config.load_checkpoint_path = '/path/to/checkpoint'

    trainer = Trainer(train_config)

    with mock.patch.object(logger, 'error') as mock_error:
        start_epoch = trainer._load_checkpoint_if_any()

        assert start_epoch == 1
        mock_isfile.assert_called_with('/path/to/checkpoint')
        mock_error.assert_called()
