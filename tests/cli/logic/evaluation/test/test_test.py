import logging
from unittest.mock import MagicMock, patch

import pytest

from advsecurenet.shared.types.configs.configs import ConfigType
from cli.logic.evaluation.test.test import cli_test
from cli.shared.types.evaluation.testing import TestingCliConfigType

logger = logging.getLogger(__name__)


@patch("cli.logic.evaluation.test.test.load_and_instantiate_config")
@patch("cli.logic.evaluation.test.test.CLITester")
def test_cli_test_success(mock_CLITester, mock_load_and_instantiate_config):
    mock_config_data = MagicMock()
    mock_load_and_instantiate_config.return_value = mock_config_data
    mock_tester_instance = mock_CLITester.return_value

    cli_test("config_path", extra_arg="value")

    mock_load_and_instantiate_config.assert_called_once_with(
        config="config_path",
        default_config_file="test_config.yml",
        config_type=ConfigType.TEST,
        config_class=TestingCliConfigType,
        extra_arg="value"
    )
    mock_CLITester.assert_called_once_with(mock_config_data)
    mock_tester_instance.test.assert_called_once()
    logger.info("Model testing completed successfully")


@patch("cli.logic.evaluation.test.test.load_and_instantiate_config")
@patch("cli.logic.evaluation.test.test.CLITester")
def test_cli_test_test_failure(mock_CLITester, mock_load_and_instantiate_config):
    mock_config_data = MagicMock()
    mock_load_and_instantiate_config.return_value = mock_config_data
    mock_tester_instance = mock_CLITester.return_value
    test_error = Exception("Test error")
    mock_tester_instance.test.side_effect = test_error

    with patch("logging.error") as mock_logging_error:
        with pytest.raises(Exception, match="Test error"):
            cli_test("config_path", extra_arg="value")

        mock_load_and_instantiate_config.assert_called_once_with(
            config="config_path",
            default_config_file="test_config.yml",
            config_type=ConfigType.TEST,
            config_class=TestingCliConfigType,
            extra_arg="value"
        )
        mock_CLITester.assert_called_once_with(mock_config_data)
        mock_tester_instance.test.assert_called_once()
        mock_logging_error.assert_called_once_with(
            "Failed to test model: %s", test_error)
