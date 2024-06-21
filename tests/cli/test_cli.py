import pytest
from click.testing import CliRunner

from cli import __version__ as version
from cli.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.cli
@pytest.mark.essential
def test_main_help(runner):
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0
    assert 'Usage:' in result.output
    assert 'Welcome to AdvSecureNet CLI!' in result.output


@pytest.mark.cli
@pytest.mark.essential
def test_main_version(runner):
    result = runner.invoke(main, ['--version'])
    assert result.exit_code == 0
    assert str(version) in result.output


@pytest.mark.cli
@pytest.mark.essential
def test_attack_command(runner):
    result = runner.invoke(main, ['attack', '--help'])
    assert result.exit_code == 0
    assert 'Usage:' in result.output


@pytest.mark.cli
@pytest.mark.essential
def test_defense_command(runner):
    result = runner.invoke(main, ['defense', '--help'])
    assert result.exit_code == 0
    assert 'Usage:' in result.output


@pytest.mark.cli
@pytest.mark.essential
def test_evaluate_command(runner):
    result = runner.invoke(main, ['evaluate', '--help'])
    assert result.exit_code == 0
    assert 'Usage:' in result.output


@pytest.mark.cli
@pytest.mark.essential
def test_utils_command(runner):
    result = runner.invoke(main, ['utils', '--help'])
    assert result.exit_code == 0
    assert 'Usage:' in result.output


@pytest.mark.cli
@pytest.mark.essential
def test_train_command(runner):
    result = runner.invoke(main, ['train', '--help'])
    assert result.exit_code == 0
    assert 'Usage:' in result.output
