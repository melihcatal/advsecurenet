import importlib.util
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from advsecurenet.models.external_model import ExternalModel
from advsecurenet.shared.types.configs.model_config import ExternalModelConfig


# Mock configuration
@pytest.fixture
def mock_config():
    config = ExternalModelConfig(
        model_name='MockModel',
        model_arch_path='/path/to/mock_model.py',
        pretrained=False,
        model_weights_path='/path/to/mock_model_weights.pth'
    )
    return config


class MockModelClass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.layer(x)


@pytest.fixture
def mock_model_class():
    return MockModelClass


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('os.path.exists', return_value=True)
@patch('importlib.util.spec_from_file_location')
@patch('importlib.util.module_from_spec')
def test_load_model(mock_spec, mock_module_from_spec, mock_exists, mock_config, mock_model_class):
    # Mocking the importlib functionality
    spec_mock = MagicMock()
    mock_spec.return_value = spec_mock
    module_mock = MagicMock()
    setattr(module_mock, 'MockModel', mock_model_class)
    mock_module_from_spec.return_value = module_mock

    # Initialize ExternalModel
    external_model = ExternalModel(config=mock_config)

    # Assertions
    mock_exists.assert_called_once_with('/path/to/mock_model.py')
    assert isinstance(external_model.model, MagicMock)


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('os.path.exists', return_value=False)
def test_load_model_file_not_found(mock_exists, mock_config):
    with pytest.raises(FileNotFoundError):
        external_model = ExternalModel(config=mock_config)
        external_model.load_model()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('os.path.exists', return_value=True)
@patch('importlib.util.spec_from_file_location')
@patch('importlib.util.module_from_spec')
@patch('advsecurenet.models.external_model.ExternalModel._ensure_model_class_exists', side_effect=ValueError)
def test_load_model_class_not_found(mock_module_from_spec, mock_spec, mock_exists, mock_config):
    # Mocking the importlib functionality
    spec_mock = MagicMock()
    mock_spec.return_value = spec_mock

    # Create a mock module without the 'NonExistentModel' class
    module_mock = MagicMock()
    mock_module_from_spec.return_value = module_mock

    with pytest.raises(ValueError):
        # Initialize ExternalModel
        external_model = ExternalModel(config=mock_config)
        external_model.load_model()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('os.path.exists', return_value=True)
@patch('importlib.util.spec_from_file_location')
@patch('importlib.util.module_from_spec')
@patch('torch.load')
def test_load_model_with_pretrained(mock_torch_load, mock_spec, mock_module_from_spec, mock_exists, mock_config, mock_model_class):
    # Mocking the importlib functionality
    spec_mock = MagicMock()
    mock_spec.return_value = spec_mock
    module_mock = MagicMock()
    setattr(module_mock, 'MockModel', mock_model_class)
    mock_module_from_spec.return_value = module_mock

    # Update config to use pretrained weights
    mock_config.pretrained = True

    # Initialize ExternalModel
    external_model = ExternalModel(config=mock_config)

    # Assertions
    mock_exists.assert_called_once_with('/path/to/mock_model.py')
    mock_torch_load.assert_called_once_with('/path/to/mock_model_weights.pth')
    print(
        f"external model model: {external_model.model} type: {type(external_model.model)}")
    assert isinstance(external_model.model, MagicMock)
