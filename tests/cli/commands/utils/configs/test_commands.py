from unittest.mock import patch

import pytest
from click.testing import CliRunner

from cli.commands.utils.commands import configs


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.cli
@pytest.mark.essential
@patch('cli.logic.utils.config.cli_config_default')
def test_get_command_with_options(mock_cli_config_default, runner):
    result = runner.invoke(configs, [
                           'get', '-c', 'train', '-s', '-p', '-o', './myconfigs/mytrain_config.yml'])
    assert result.exit_code == 0
    mock_cli_config_default.assert_called_once_with(
        'train', True, True, './myconfigs/mytrain_config.yml')


@pytest.mark.cli
@pytest.mark.essential
@patch('cli.logic.utils.config.cli_config_default')
def test_get_command_without_options(mock_cli_config_default, runner):
    result = runner.invoke(configs, ['get'])
    assert result.exit_code == 0
    mock_cli_config_default.assert_called_once_with(
        None, False, False, None)


@pytest.mark.cli
@pytest.mark.essential
@patch('cli.logic.utils.config.cli_configs')
def test_list_command(mock_cli_configs, runner):
    result = runner.invoke(configs, ['list'])
    assert result.exit_code == 0
    mock_cli_configs.assert_called_once()
