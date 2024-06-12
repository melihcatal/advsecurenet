from unittest.mock import patch

import pytest
from click.testing import CliRunner

from cli.commands.utils.commands import utils


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.cli
@pytest.mark.essential
@patch('cli.logic.utils.normalization_params.cli_normalization_params')
def test_normalization_params_command(mock_cli_normalization_params, runner):
    result = runner.invoke(
        utils, ['normalization-params', '--dataset-name', 'CIFAR10'])
    assert result.exit_code == 0
    mock_cli_normalization_params.assert_called_once_with('CIFAR10')


@pytest.mark.cli
@pytest.mark.essential
@patch('cli.logic.utils.normalization_params.cli_normalization_params')
def test_normalization_params_command_no_dataset_name(mock_cli_normalization_params, runner):
    result = runner.invoke(utils, ['normalization-params'])
    assert result.exit_code == 0
    mock_cli_normalization_params.assert_called_once_with(None)
