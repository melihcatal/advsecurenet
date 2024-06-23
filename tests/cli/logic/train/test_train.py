import logging
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

from advsecurenet.shared.types.configs.configs import ConfigType
from cli.logic.train.train import cli_train
from cli.shared.types.train import TrainingCliConfigType
from cli.shared.utils.config import make_paths_absolute

logger = logging.getLogger("cli.logic.train.train")


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.train.train.load_and_instantiate_config")
@patch("cli.logic.train.train.CLITrainer")
def test_cli_train_success(mock_CLITrainer, mock_load_and_instantiate_config):
    mock_config_data = MagicMock()
    mock_load_and_instantiate_config.return_value = mock_config_data
    mock_trainer_instance = mock_CLITrainer.return_value

    cli_train("config_path", extra_arg="value")

    mock_load_and_instantiate_config.assert_called_once_with(
        "config_path", "train_config.yml", ConfigType.TRAIN, TrainingCliConfigType, extra_arg="value"
    )
    mock_CLITrainer.assert_called_once_with(mock_config_data)
    mock_trainer_instance.train.assert_called_once()
    logger.info("Training completed successfully")


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.train.train.load_and_instantiate_config")
@patch("cli.logic.train.train.CLITrainer")
def test_cli_train_train_failure(mock_CLITrainer, mock_load_and_instantiate_config):
    mock_config_data = MagicMock()
    mock_load_and_instantiate_config.return_value = mock_config_data
    mock_trainer_instance = mock_CLITrainer.return_value
    training_error = Exception("Training error")
    mock_trainer_instance.train.side_effect = training_error

    with mock.patch.object(logger, 'error') as mock_logging_error:
        with pytest.raises(Exception, match="Training error"):
            cli_train("config_path", extra_arg="value")

        mock_load_and_instantiate_config.assert_called_once_with(
            "config_path", "train_config.yml", ConfigType.TRAIN, TrainingCliConfigType, extra_arg="value"
        )
        mock_CLITrainer.assert_called_once_with(mock_config_data)
        mock_trainer_instance.train.assert_called_once()
        mock_logging_error.assert_called_once_with(
            "Failed to train model: %s", training_error)
