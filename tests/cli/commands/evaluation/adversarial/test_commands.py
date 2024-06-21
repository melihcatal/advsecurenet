from unittest.mock import patch

import pytest
from click.testing import CliRunner

from cli.commands.evaluation.adversarial.commands import adversarial


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.cli
@pytest.mark.essential
def test_adversarial_help(runner):
    result = runner.invoke(adversarial, ['--help'])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Command to evaluate models on adversarial examples." in result.output


@pytest.mark.cli
@pytest.mark.essential
def test_adversarial_list_help(runner):
    result = runner.invoke(adversarial, ['list', '--help'])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Command to list available adversarial evaluation options." in result.output


@pytest.mark.cli
@pytest.mark.essential
@patch('cli.logic.evaluation.evaluation.cli_list_adversarial_evaluations')
def test_adversarial_list(mock_cli_list_adversarial_evaluations, runner):
    result = runner.invoke(adversarial, ['list'])
    assert result.exit_code == 0
    mock_cli_list_adversarial_evaluations.assert_called_once()


@pytest.mark.cli
@pytest.mark.essential
def test_adversarial_eval_help(runner):
    result = runner.invoke(adversarial, ['eval', '--help'])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Command to evaluate the model on adversarial examples." in result.output


@pytest.mark.cli
@pytest.mark.essential
@patch('cli.logic.evaluation.evaluation.cli_adversarial_evaluation')
def test_adversarial_eval(mock_cli_adversarial_evaluation, runner):
    with runner.isolated_filesystem():
        with open('eval_config.yml', 'w') as f:
            f.write(
                'model_name: resnet18\ndataset_name: cifar10\nmodel_weights: resnet18_cifar10_weights.pth')
        result = runner.invoke(
            adversarial, ['eval', '--config', 'eval_config.yml'])
        assert result.exit_code == 0
        mock_cli_adversarial_evaluation.assert_called_once_with(
            'eval_config.yml')
