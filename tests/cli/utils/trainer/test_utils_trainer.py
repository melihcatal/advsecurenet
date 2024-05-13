import copy
import os
from unittest.mock import Mock, patch

import pytest

from advsecurenet.shared.types.configs import ConfigType
from advsecurenet.shared.types.dataset import DatasetType
from cli.logic.train.trainer import CLITrainer
from cli.shared.utils.config import load_configuration
from cli.types.train.training import TrainingCliConfigType

# Replace 'your_module' with the actual name of the module where CLITrainer is defined.

# Test for _validate_dataset_name method


@pytest.fixture(scope="module")
def set_working_dir():
    # Save the original working directory
    original_dir = os.getcwd()
    # Get the path to the 'tests/cli' directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print("current_dir: ", current_dir)
    cli_dir = os.path.join(current_dir, '..')
    os.chdir(current_dir)
    yield
    # Reset the working directory after the test
    os.chdir(original_dir)


@pytest.fixture(scope="module")
def config(set_working_dir) -> TrainingCliConfigType:
    config_path = "./train_config.yml"
    config_data = load_configuration(
        config_type=ConfigType.TRAIN, config_file=config_path)
    config_data = TrainingCliConfigType(**config_data)
    return config_data


def test_validate_dataset_name_valid(config):
    trainer = CLITrainer(config)
    assert trainer._validate_dataset_name() == "CIFAR10"


def test_validate_dataset_name_invalid(config):
    config.dataset_name = "INVALID_DATASET"
    with pytest.raises(ValueError):
        trainer = CLITrainer(config)

# Test for _validate_config method


def test_validate_config_valid(config):
    # mock the _initialize_params method
    mock_init_params = Mock()
    mock_init_params.return_value = None
    CLITrainer._initialize_params = mock_init_params
    trainer = CLITrainer(config)
    # If no exception is raised, the test passes
    trainer._validate_config(config)


def test_validate_config_invalid_no_model_name(config):
    mock_config = copy.deepcopy(config)
    mock_config.model_name = None
    with pytest.raises(ValueError):
        trainer = CLITrainer(mock_config)
        trainer._validate_config(mock_config)


def test_validate_config_invalid_no_dataset_name(config):
    mock_config = copy.deepcopy(config)
    mock_config.dataset_name = None
    with pytest.raises(ValueError):
        trainer = CLITrainer(mock_config)
        trainer._validate_config(mock_config)


@patch('cli.shared.utils.trainer.DatasetFactory.create_dataset')
@patch('cli.shared.utils.trainer.CLITrainer._validate_dataset_name', return_value="CIFAR10")
def test_load_datasets(mock_validate, mock_create_dataset, config):
    trainer = CLITrainer(config)

    # Mock dataset objects
    mock_train_dataset = Mock()
    mock_test_dataset = Mock()
    mock_dataset_obj = Mock()
    mock_dataset_obj.load_dataset.side_effect = [
        mock_train_dataset, mock_test_dataset]

    mock_create_dataset.return_value = mock_dataset_obj

    train_data, test_data = trainer._load_datasets(
        "CIFAR10")

    assert train_data == mock_train_dataset
    assert test_data == mock_test_dataset
