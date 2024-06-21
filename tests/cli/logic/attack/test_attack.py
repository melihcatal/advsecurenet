import logging
from unittest.mock import MagicMock, patch

import pytest

from advsecurenet.shared.types.configs.configs import ConfigType
from cli.logic.attack.attack import cli_attack
from cli.logic.attack.attacker import CLIAttacker
from cli.shared.types.attack import BaseAttackCLIConfigType
from cli.shared.utils.attack_mappings import attack_cli_mapping
from cli.shared.utils.config import load_and_instantiate_config

# Configure logger for testing
logger = logging.getLogger('cli_attack')
logger.setLevel(logging.DEBUG)


@pytest.fixture
def mock_attack_cli_mapping():
    with patch.dict(attack_cli_mapping, {'test_attack': ('attack_type', BaseAttackCLIConfigType)}):
        yield


@pytest.fixture
def mock_load_and_instantiate_config():
    with patch('cli.logic.attack.attack.load_and_instantiate_config', return_value=MagicMock(spec=BaseAttackCLIConfigType)) as mock:
        yield mock


@pytest.fixture
def mock_cliatacker():
    with patch('cli.logic.attack.attack.CLIAttacker') as mock:
        yield mock


@pytest.mark.cli
@pytest.mark.essential
def test_cli_attack_success(mock_attack_cli_mapping, mock_load_and_instantiate_config, mock_cliatacker):
    # Mocking successful execution
    mock_instance = mock_cliatacker.return_value
    mock_instance.execute.return_value = None

    cli_attack('test_attack', 'config_path.yml')

    mock_load_and_instantiate_config.assert_called_once_with(
        config='config_path.yml',
        default_config_file='test_attack_attack_config.yml',
        config_type=ConfigType.ATTACK,
        config_class=BaseAttackCLIConfigType
    )
    mock_cliatacker.assert_called_once_with(
        mock_load_and_instantiate_config.return_value, 'attack_type')
    mock_instance.execute.assert_called_once()
    assert logger.hasHandlers() is True


@pytest.mark.cli
@pytest.mark.essential
def test_cli_attack_unknown_attack():
    with pytest.raises(ValueError, match="Unknown attack type: unknown_attack"):
        cli_attack('unknown_attack', 'config_path.yml')


@pytest.mark.cli
@pytest.mark.essential
def test_cli_attack_execution_failure(mock_attack_cli_mapping, mock_load_and_instantiate_config, mock_cliatacker):
    # Mocking execution failure
    mock_instance = mock_cliatacker.return_value
    mock_instance.execute.side_effect = Exception("Execution error")

    with pytest.raises(Exception, match="Execution error"):
        cli_attack('test_attack', 'config_path.yml')

    mock_load_and_instantiate_config.assert_called_once_with(
        config='config_path.yml',
        default_config_file='test_attack_attack_config.yml',
        config_type=ConfigType.ATTACK,
        config_class=BaseAttackCLIConfigType
    )
    mock_cliatacker.assert_called_once_with(
        mock_load_and_instantiate_config.return_value, 'attack_type')
    mock_instance.execute.assert_called_once()
