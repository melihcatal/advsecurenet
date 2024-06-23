import logging
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

from advsecurenet.shared.types.configs.configs import ConfigType
from cli.logic.defense.defense import cli_adversarial_training
from cli.shared.types.defense.adversarial_training import ATCliConfigType

logger = logging.getLogger("cli.logic.defense.defense")


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.defense.defense.load_and_instantiate_config")
@patch("cli.logic.defense.defense.ATCLITrainer")
def test_cli_adversarial_training_success(mock_ATCLITrainer, mock_load_and_instantiate_config):
    mock_config_data = MagicMock()
    mock_load_and_instantiate_config.return_value = mock_config_data
    mock_trainer_instance = mock_ATCLITrainer.return_value

    cli_adversarial_training("config_path", extra_arg="value")

    mock_load_and_instantiate_config.assert_called_once_with(
        config="config_path",
        default_config_file="adversarial_training_config.yml",
        config_type=ConfigType.ADVERSARIAL_TRAINING,
        config_class=ATCliConfigType,
        extra_arg="value"
    )
    mock_ATCLITrainer.assert_called_once_with(mock_config_data)
    mock_trainer_instance.train.assert_called_once()
    logger.info("Adversarial training completed successfully")


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.defense.defense.load_and_instantiate_config")
@patch("cli.logic.defense.defense.ATCLITrainer")
def test_cli_adversarial_training_train_failure(mock_ATCLITrainer, mock_load_and_instantiate_config):
    mock_config_data = MagicMock()
    mock_load_and_instantiate_config.return_value = mock_config_data
    mock_trainer_instance = mock_ATCLITrainer.return_value
    training_error = Exception("Training error")
    mock_trainer_instance.train.side_effect = training_error

    with mock.patch.object(logger, 'error') as mock_logging_error:
        with pytest.raises(Exception, match="Training error"):
            cli_adversarial_training("config_path", extra_arg="value")

        mock_load_and_instantiate_config.assert_called_once_with(
            config="config_path",
            default_config_file="adversarial_training_config.yml",
            config_type=ConfigType.ADVERSARIAL_TRAINING,
            config_class=ATCliConfigType,
            extra_arg="value"
        )
        mock_ATCLITrainer.assert_called_once_with(mock_config_data)
        mock_trainer_instance.train.assert_called_once()
        mock_logging_error.assert_called_once_with(
            "Failed to execute adversarial training: %s", training_error)
