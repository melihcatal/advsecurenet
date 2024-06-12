from enum import EnumMeta
from unittest.mock import MagicMock, patch

import pytest
from torch import nn

from advsecurenet.models import CustomModel, StandardModel
from advsecurenet.models.base_model import BaseModel
from advsecurenet.models.CustomModels.CustomMnistModel import CustomMnistModel
from advsecurenet.models.external_model import ExternalModel
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types.configs.model_config import (
    CreateModelConfig, CustomModelConfig, ExternalModelConfig,
    StandardModelConfig)
from advsecurenet.shared.types.model import ModelType
from advsecurenet.utils.reproducibility_utils import set_seed


@pytest.fixture
def create_model_config():
    return CreateModelConfig(
        model_name="resnet18",
        num_classes=10,
        pretrained=True,
        weights="IMAGENET1K_V1",
        random_seed=42
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_infer_model_type_standard():
    with patch.object(StandardModel, 'models', return_value=['resnet18']):
        model_type = ModelFactory.infer_model_type('resnet18')
        assert model_type == ModelType.STANDARD


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_infer_model_type_custom():
    with patch.object(CustomModel, 'models', return_value=['CustomMnistModel']):
        model_type = ModelFactory.infer_model_type('CustomMnistModel')
        assert model_type == ModelType.CUSTOM


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_infer_model_type_invalid():
    with patch.object(StandardModel, 'models', return_value=[]), \
            patch.object(CustomModel, 'models', return_value=[]):
        with pytest.raises(ValueError, match="Unsupported model"):
            ModelFactory.infer_model_type('invalid_model')


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch.object(StandardModel, 'models', return_value=['resnet18'])
@patch('advsecurenet.models.StandardModel')
def test_create_model_standard(mock_standard_model, create_model_config):
    cfg = CreateModelConfig(
        model_name="resnet18",
        num_classes=10,
        pretrained=True,
        weights="IMAGENET1K_V1"
    )
    model = ModelFactory.create_model(config=cfg)
    assert isinstance(model, BaseModel)
    assert model._model_name == "resnet18"


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.models.ExternalModel')
@patch('os.path.exists', return_value=True)
@patch('importlib.util.spec_from_file_location')
@patch('importlib.util.module_from_spec')
def test_create_model_external(mock_module_from_spec, mock_spec_from_file_location, mock_exists, mock_external_model):
    config = CreateModelConfig(
        model_name='MockExternalModel',
        num_classes=10,
        model_arch_path='/path/to/mock_model.py',
        pretrained=False,
        is_external=True
    )

    # Mock the module loading
    spec_mock = MagicMock()
    mock_spec_from_file_location.return_value = spec_mock
    module_mock = MagicMock()
    mock_module_from_spec.return_value = module_mock

    # Mock the external model class in the module
    setattr(module_mock, 'MockExternalModel', MagicMock(spec=BaseModel))

    # Mock the return value of the ExternalModel constructor
    model_instance = MagicMock(spec=BaseModel)
    mock_external_model.return_value = model_instance

    created_model = ModelFactory.create_model(config=config)

    mock_exists.assert_called_with('/path/to/mock_model.py')
    mock_spec_from_file_location.assert_called_once_with(
        'MockExternalModel', '/path/to/mock_model.py')
    mock_module_from_spec.assert_called_once_with(spec_mock)
    assert isinstance(created_model, BaseModel)


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch.object(CustomModel, 'models', return_value=['CustomMnistModel'])
@patch('advsecurenet.models.CustomModel')
def test_create_model_custom(mock_custom_model, mock_models):
    config = CreateModelConfig(
        model_name='CustomMnistModel',
        num_classes=10,
        num_input_channels=1,
        pretrained=False
    )
    model = CustomMnistModel()
    mock_custom_model.return_value = model

    created_model = ModelFactory.create_model(config=config)

    assert isinstance(created_model.model, CustomMnistModel)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_validate_create_model_config():
    config = CreateModelConfig(
        model_name='resnet18',
        num_classes=10,
        pretrained=True,
        random_seed=42
    )
    with pytest.raises(ValueError, match="Pretrained standard models do not support random seed"):
        ModelFactory._validate_create_model_config(ModelType.STANDARD, config)

    config = CreateModelConfig(
        model_name='CustomMnistModel',
        num_classes=10,
        pretrained=True
    )
    with pytest.raises(ValueError, match="Custom models do not support pretrained weights"):
        ModelFactory._validate_create_model_config(ModelType.CUSTOM, config)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_add_layer():
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU()
    )
    new_layer = nn.Linear(20, 10)
    updated_model = ModelFactory.add_layer(model, new_layer, 1)
    assert isinstance(updated_model, nn.Sequential)
    assert len(updated_model) == 3
    assert isinstance(updated_model[1], nn.Linear)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_available_models():
    with patch.object(StandardModel, 'models', return_value=['resnet18']), \
            patch.object(CustomModel, 'models', return_value=['CustomMnistModel']):
        available_models = ModelFactory.available_models()
        assert available_models == ['resnet18', 'CustomMnistModel']


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_available_standard_models():
    with patch.object(StandardModel, 'models', return_value=['resnet18']):
        available_models = ModelFactory.available_standard_models()
        assert available_models == ['resnet18']


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_available_custom_models():
    with patch.object(CustomModel, 'models', return_value=['CustomMnistModel']):
        available_models = ModelFactory.available_custom_models()
        assert available_models == ['CustomMnistModel']


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_available_weights():
    with patch.object(StandardModel, 'models', return_value=['resnet18']), \
            patch.object(StandardModel, 'available_weights', return_value=MagicMock(spec=EnumMeta)):
        weights = ModelFactory.available_weights('resnet18')
        assert isinstance(weights, EnumMeta)
