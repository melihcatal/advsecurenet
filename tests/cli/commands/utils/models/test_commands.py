from unittest.mock import patch

import pytest
from click.testing import CliRunner

from cli.commands.utils.models.commands import models


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.cli
@pytest.mark.essential
@patch('cli.logic.utils.model.cli_models')
def test_list_command(mock_cli_models, runner):
    result = runner.invoke(models, ['list', '--model-type', 'custom'])
    assert result.exit_code == 0
    mock_cli_models.assert_called_once_with('custom')


@pytest.mark.cli
@pytest.mark.essential
@patch('cli.logic.utils.model.cli_models')
def test_list_command_default(mock_cli_models, runner):
    result = runner.invoke(models, ['list'])
    assert result.exit_code == 0
    mock_cli_models.assert_called_once_with('all')


@pytest.mark.cli
@pytest.mark.essential
@patch('cli.logic.utils.model.cli_model_layers')
def test_layers_command(mock_cli_model_layers, runner):
    result = runner.invoke(
        models, ['layers', '--model-name', 'resnet18', '--normalization'])
    assert result.exit_code == 0
    mock_cli_model_layers.assert_called_once_with('resnet18', True)


@pytest.mark.cli
@pytest.mark.essential
@patch('cli.logic.utils.model.cli_model_layers')
def test_layers_command_no_normalization(mock_cli_model_layers, runner):
    result = runner.invoke(models, ['layers', '--model-name', 'resnet18'])
    assert result.exit_code == 0
    mock_cli_model_layers.assert_called_once_with('resnet18', False)
