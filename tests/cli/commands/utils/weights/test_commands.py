from unittest.mock import patch

import pytest
from click.testing import CliRunner

from cli.commands.utils.weights.commands import weights


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.cli
@pytest.mark.essential
@patch('cli.logic.utils.model.cli_available_weights')
def test_list_command(mock_cli_available_weights, runner):
    result = runner.invoke(weights, ['list', '--model-name', 'resnet18'])
    assert result.exit_code == 0
    mock_cli_available_weights.assert_called_once_with('resnet18')


@pytest.mark.cli
@pytest.mark.essential
@patch('cli.logic.utils.model.cli_download_weights')
def test_download_command(mock_cli_download_weights, runner):
    result = runner.invoke(weights, [
        'download',
        '--model-name', 'resnet18',
        '--dataset-name', 'cifar10',
        '--filename', 'resnet18_weights.pth',
        '--save-path', './weights'
    ])
    assert result.exit_code == 0
    mock_cli_download_weights.assert_called_once_with(
        'resnet18', 'cifar10', 'resnet18_weights.pth', './weights'
    )


@pytest.mark.cli
@pytest.mark.essential
@patch('cli.logic.utils.model.cli_download_weights')
def test_download_command_defaults(mock_cli_download_weights, runner):
    result = runner.invoke(weights, ['download', '--model-name', 'resnet18'])
    assert result.exit_code == 0
    mock_cli_download_weights.assert_called_once_with(
        'resnet18', None, None, None
    )
