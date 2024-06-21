import pytest
from click.testing import CliRunner

from cli.commands.defense import defense


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.cli
@pytest.mark.essential
def test_defense_adversarial_training_help(runner):
    result = runner.invoke(defense, ['adversarial-training', '--help'])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Command to execute adversarial training." in result.output
