from unittest.mock import MagicMock, patch

import pytest

from cli.logic.train.trainer import CLITrainer
from cli.shared.types.train import TrainingCliConfigType


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.train.trainer.CLITrainer._prepare_dataset")
@patch("cli.logic.train.trainer.create_model")
@patch("cli.logic.train.trainer.get_dataloader")
def test_cli_trainer_init(mock_prepare_dataset, mock_create_model, mock_get_dataloader):
    # Setup
    mock_config = MagicMock()
    mock_train_data = MagicMock()
    mock_prepare_dataset.return_value = mock_train_data

    # Initialize CLITrainer
    trainer = CLITrainer(mock_config)

    # Verify
    assert trainer.config == mock_config


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.train.trainer.CLITrainer._prepare_dataset", return_value=MagicMock())
@patch("cli.logic.train.trainer.set_visible_gpus")
@patch("cli.logic.train.trainer.DDPCoordinator")
@patch("cli.logic.train.trainer.DDPTrainer")
def test_cli_trainer_execute_ddp_training(mock_ddp_trainer, mock_ddp_training_coordinator, mock_set_visible_gpus, mock_prepare_dataset):
    # Setup
    mock_config = MagicMock()
    mock_config.device.use_ddp = True
    mock_config.device.gpu_ids = [0, 1]
    trainer = CLITrainer(mock_config)

    # Mock methods
    trainer._prepare_training_environment = MagicMock(return_value=MagicMock())

    # Execute DDP training
    trainer._execute_ddp_training()

    # Verify
    mock_set_visible_gpus.assert_called_once_with([0, 1])
    mock_ddp_training_coordinator.assert_called_once_with(
        trainer._ddp_training_fn, 2)
    mock_ddp_training_coordinator.return_value.run.assert_called_once()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.train.trainer.CLITrainer._prepare_training_environment", return_value=MagicMock(spec=TrainingCliConfigType))
@patch("cli.logic.train.trainer.CLITrainer._prepare_dataset", return_value=MagicMock())
@patch("cli.logic.train.trainer.Trainer")
def test_cli_trainer_execute_training(mock_trainer, mock_prepare_dataset, mock_prepare_training_environment):
    # Setup
    mock_config = MagicMock()
    mock_config.device.use_ddp = False
    trainer = CLITrainer(mock_config)

    # Mock methods
    mock_train_config = MagicMock()
    trainer._prepare_training_environment = MagicMock(
        return_value=mock_train_config)

    # Execute training
    trainer._execute_training()

    # Verify
    mock_trainer.assert_called_once_with(mock_train_config)
    mock_trainer.return_value.train.assert_called_once()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.train.trainer.CLITrainer._prepare_dataset", return_value=MagicMock())
@patch("cli.logic.train.trainer.create_model")
def test_cli_trainer_initialize_model(mock_create_model, mock_prepare_dataset):
    # Setup
    mock_config = MagicMock()
    trainer = CLITrainer(mock_config)

    # Execute
    trainer._initialize_model()

    # Verify
    mock_create_model.assert_called_once_with(mock_config.model)


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.train.trainer.get_dataloader")
@patch("cli.logic.train.trainer.get_datasets")
def test_cli_trainer_prepare_dataloader(mock_get_datasets, mock_get_dataloader):
    # Setup
    mock_config = MagicMock()
    mock_train_data = MagicMock()
    mock_get_datasets.return_value = (mock_train_data, None)
    trainer = CLITrainer(mock_config)

    # Execute
    dataloader = trainer._prepare_dataloader()

    # Verify
    mock_get_dataloader.assert_called_once_with(
        config=mock_config.dataloader,
        dataset=mock_train_data,
        dataset_type='train',
        use_ddp=mock_config.device.use_ddp
    )
    assert dataloader == mock_get_dataloader.return_value


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.train.trainer.CLITrainer._prepare_dataset", return_value=MagicMock())
@patch("cli.logic.train.trainer.DDPTrainer")
def test_cli_trainer_ddp_training_fn(mock_ddp_trainer, mock_prepare_dataset):
    # Setup
    mock_config = MagicMock()
    trainer = CLITrainer(mock_config)

    # Mock methods
    mock_train_config = MagicMock()
    trainer._prepare_training_environment = MagicMock(
        return_value=mock_train_config)

    # Execute
    trainer._ddp_training_fn(0, 2)

    # Verify
    mock_ddp_trainer.assert_called_once_with(mock_train_config, 0, 2)
    mock_ddp_trainer.return_value.train.assert_called_once()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.train.trainer.CLITrainer._prepare_dataset", return_value=MagicMock())
def test_cli_trainer_execute_with_exception(mock_prepare_dataset):
    # Setup
    mock_config = MagicMock()
    mock_config.device.use_ddp = False
    trainer = CLITrainer(mock_config)

    # Mock methods
    trainer._execute_training = MagicMock(
        side_effect=Exception("Training error"))

    # Execute and verify exception
    with pytest.raises(Exception, match="Training error"):
        trainer.train()

    trainer._execute_training.assert_called_once()
