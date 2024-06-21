from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from cli.commands.evaluation.benign.commands import benign


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.cli
@pytest.mark.essential
def test_evaluate_test_help(runner):
    result = runner.invoke(benign, ['test', '--help'])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Command to test a model on a dataset." in result.output


@pytest.mark.cli
@pytest.mark.essential
@patch('cli.logic.evaluation.test.test.cli_test')
def test_evaluate_test(mock_cli_test, runner):
    result = runner.invoke(benign, ['test', '--model-name', 'resnet18',
                           '--dataset-name', 'cifar10', '--model-weights', 'resnet18_cifar10_weights.pth'])
    assert result.exit_code == 0
    mock_cli_test.assert_called_once_with(None,
                                          model_name='resnet18',
                                          dataset_name='cifar10',
                                          model_weights='resnet18_cifar10_weights.pth',
                                          processor=None,
                                          batch_size=None,
                                          loss=None,
                                          topk=None
                                          )


@pytest.mark.cli
@pytest.mark.essential
@patch('cli.logic.evaluation.test.test.cli_test')
@patch('click.Path', return_value=MagicMock(exists=True))
def test_evaluate_test_with_config(mock_click_path, mock_cli_test, runner):
    with runner.isolated_filesystem():
        with open('test_config.yml', 'w') as f:
            f.write(
                'model_name: resnet18\ndataset_name: cifar10\nmodel_weights: resnet18_cifar10_weights.pth')
        result = runner.invoke(
            benign, ['test', '--config', 'test_config.yml'])
        assert result.exit_code == 0
        mock_cli_test.assert_called_once_with(
            'test_config.yml',
            model_name=None,
            dataset_name=None,
            model_weights=None,
            processor=None,
            batch_size=None,
            loss=None,
            topk=None
        )
