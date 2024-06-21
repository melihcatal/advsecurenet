import pytest
from click.testing import CliRunner

from cli.commands.attack import attack


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.cli
@pytest.mark.essential
def test_attack_deepfool_help(runner):
    result = runner.invoke(attack, ['deepfool', '--help'])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Command to execute a DeepFool attack." in result.output


@pytest.mark.cli
@pytest.mark.essential
def test_attack_cw_help(runner):
    result = runner.invoke(attack, ['cw', '--help'])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Command to execute a Carlini-Wagner attack." in result.output


@pytest.mark.cli
@pytest.mark.essential
def test_attack_pgd_help(runner):
    result = runner.invoke(attack, ['pgd', '--help'])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Command to execute a PGD attack." in result.output


@pytest.mark.cli
@pytest.mark.essential
def test_attack_fgsm_help(runner):
    result = runner.invoke(attack, ['fgsm', '--help'])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Command to execute a FGSM attack." in result.output


@pytest.mark.cli
@pytest.mark.essential
def test_attack_decision_boundary_help(runner):
    result = runner.invoke(attack, ['decision-boundary', '--help'])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Command to execute a Decision Boundary attack." in result.output


@pytest.mark.cli
@pytest.mark.essential
def test_attack_lots_help(runner):
    result = runner.invoke(attack, ['lots', '--help'])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Command to execute a LOTS attack." in result.output
