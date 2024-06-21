from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from cli.commands.train.commands import INT_LIST, train


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.cli
@pytest.mark.essential
@patch('cli.logic.train.train.cli_train')
def test_train_with_config(mock_cli_train, runner):
    with runner.isolated_filesystem():
        with open('train_config.yml', 'w') as f:
            f.write('model_name: resnet18\n')
            f.write('dataset_name: cifar10\n')

        result = runner.invoke(train, ['--config', 'train_config.yml'])
        assert result.exit_code == 0
        mock_cli_train.assert_called_once_with('train_config.yml', model_name=None, dataset_name=None, epochs=None, batch_size=None, lr=None, optimizer=None, loss=None, save=None, save_path=None, save_name=None,
                                               processor=None, save_checkpoint=None, checkpoint_interval=None, save_checkpoint_path=None, save_checkpoint_name=None, load_checkpoint=None, load_checkpoint_path=None, use_ddp=None, gpu_ids=None, pin_memory=None)


@pytest.mark.cli
@pytest.mark.essential
@patch('cli.logic.train.train.cli_train')
def test_train_with_options(mock_cli_train, runner):
    result = runner.invoke(train, [
        '--model-name', 'resnet18',
        '--dataset-name', 'cifar10',
        '--epochs', '10',
        '--batch-size', '32',

    ])
    assert result.exit_code == 0
    mock_cli_train.assert_called_once_with(
        None, model_name='resnet18', dataset_name='cifar10',
        epochs=10, batch_size=32, lr=None, optimizer=None, loss=None, save=None,
        save_path=None, save_name=None, processor=None, save_checkpoint=None,
        checkpoint_interval=None, save_checkpoint_path=None, save_checkpoint_name=None,
        load_checkpoint=None, load_checkpoint_path=None, use_ddp=None, gpu_ids=None, pin_memory=None
    )


@ pytest.mark.cli
@ pytest.mark.essential
@ patch('cli.logic.train.train.cli_train')
def test_train_with_defaults(mock_cli_train, runner):
    result = runner.invoke(train, [
        '--model-name', 'resnet18',
        '--dataset-name', 'cifar10'
    ])
    assert result.exit_code == 0
    mock_cli_train.assert_called_once_with(
        None,
        model_name='resnet18',
        dataset_name='cifar10',
        epochs=None,
        batch_size=None,
        lr=None,
        optimizer=None,
        loss=None,
        save=None,
        save_path=None,
        save_name=None,
        processor=None,
        save_checkpoint=None,
        checkpoint_interval=None,
        save_checkpoint_path=None,
        save_checkpoint_name=None,
        load_checkpoint=None,
        load_checkpoint_path=None,
        use_ddp=None,
        gpu_ids=None,
        pin_memory=None
    )


def test_int_list_param_type():
    param_type = INT_LIST
    assert param_type.convert('1,2,3', None, None) == [1, 2, 3]
    with pytest.raises(click.BadParameter):
        param_type.convert('1,a,3', None, None)
