from unittest.mock import MagicMock, patch

import pytest
import torch

from advsecurenet.models import BaseModel
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types.configs.model_config import CreateModelConfig
from cli.shared.messages.errors import CLIErrorMessages
from cli.shared.types.utils.model import (ModelCliConfigType, ModelNormConfig,
                                          ModelPathConfig)
from cli.shared.utils.model import _validate_norm_layer, create_model


@pytest.fixture
def mock_config():
    return ModelCliConfigType(
        model_name="CustomModel",
        num_input_channels=3,
        num_classes=10,
        pretrained=False,
        weights="path/to/weights",
        is_external=False,
        path_configs=ModelPathConfig(
            model_arch_path="path/to/arch",
            model_weights_path="path/to/weights"),
        norm_config=ModelNormConfig(
            add_norm_layer=False, norm_mean=None, norm_std=None),
        random_seed=None
    )


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.shared.utils.model.click.secho")
@patch("cli.shared.utils.model.torch.load")
def test_create_model_load_weights(mock_torch_load, mock_click_secho, mock_config):
    mock_config.pretrained = True
    mock_model = MagicMock(spec=BaseModel)

    with patch("cli.shared.utils.model.ModelFactory.create_model", return_value=mock_model):
        model = create_model(mock_config)
        mock_click_secho.assert_called_once_with(
            "Trying to load the model weights from the provided path...", fg="yellow")
        mock_torch_load.assert_called_once_with(
            "path/to/weights", map_location=torch.device('cpu'))
        mock_model.load_state_dict.assert_called_once_with(mock_torch_load())


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.shared.utils.model.ModelFactory.create_model")
@patch("cli.shared.utils.model.NormalizationLayer")
def test_create_model_add_norm_layer(mock_NormalizationLayer, mock_create_model, mock_config):
    mock_config.model_name = "CustomCifar10Model"
    mock_config.norm_config.add_norm_layer = True
    mock_config.norm_config.norm_mean = [0.5, 0.5, 0.5]
    mock_config.norm_config.norm_std = [0.5, 0.5, 0.5]
    mock_model = MagicMock(spec=BaseModel)
    mock_create_model.return_value = mock_model

    model = create_model(mock_config)

    mock_NormalizationLayer.assert_called_once_with(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    mock_model.add_layer.assert_called_once_with(
        new_layer=mock_NormalizationLayer(), position=0, inplace=True)
    assert model == mock_model


@pytest.mark.cli
@pytest.mark.essential
def test_validate_norm_layer_missing_mean_or_std():
    mock_config = MagicMock()
    mock_config.norm_config.add_norm_layer = True
    mock_config.norm_config.norm_mean = None
    mock_config.norm_config.norm_std = [0.5, 0.5, 0.5]
    with pytest.raises(ValueError, match=CLIErrorMessages.TRAINER.value.NORM_LAYER_MISSING_MEAN_OR_STD.value):
        _validate_norm_layer(mock_config)


@pytest.mark.cli
@pytest.mark.essential
def test_validate_norm_layer_mean_or_std_not_list():
    mock_config = MagicMock()
    mock_config.norm_config.add_norm_layer = True
    mock_config.norm_config.norm_mean = "not a list"
    mock_config.norm_config.norm_std = [0.5, 0.5, 0.5]
    with pytest.raises(ValueError, match=CLIErrorMessages.TRAINER.value.NORM_LAYER_MEAN_OR_STD_NOT_LIST.value):
        _validate_norm_layer(mock_config)


@pytest.mark.cli
@pytest.mark.essential
def test_validate_norm_layer_length_mismatch_mean_and_num_input_channels():
    mock_config = MagicMock()
    mock_config.norm_config.add_norm_layer = True
    mock_config.norm_config.norm_mean = [0.5, 0.5]
    mock_config.norm_config.norm_std = [0.5, 0.5, 0.5]
    mock_config.num_input_channels = 3
    with pytest.raises(ValueError, match=CLIErrorMessages.TRAINER.value.NORM_LAYER_LENGTH_MISMATCH_MEAN_AND_NUM_INPUT_CHANNELS.value):
        _validate_norm_layer(mock_config)


@pytest.mark.cli
@pytest.mark.essential
def test_validate_norm_layer_length_mismatch_std_and_num_input_channels():
    mock_config = MagicMock()
    mock_config.norm_config.add_norm_layer = True
    mock_config.norm_config.norm_mean = [0.5, 0.5, 0.5]
    mock_config.norm_config.norm_std = [0.5, 0.5]
    mock_config.num_input_channels = 3
    with pytest.raises(ValueError, match=CLIErrorMessages.TRAINER.value.NORM_LAYER_LENGTH_MISMATCH_STD_AND_NUM_INPUT_CHANNELS.value):
        _validate_norm_layer(mock_config)


@pytest.mark.cli
@pytest.mark.essential
def test_validate_norm_layer_invalid_type_mean_and_std(mock_config):
    mock_config.norm_config.add_norm_layer = True
    mock_config.norm_config.norm_mean = torch.randn(
        mock_config.num_input_channels)
    mock_config.norm_config.norm_std = torch.randn(
        mock_config.num_input_channels)
    with pytest.raises(ValueError, match=CLIErrorMessages.TRAINER.value.NORM_LAYER_MEAN_OR_STD_NOT_LIST.value):
        _validate_norm_layer(mock_config)


@pytest.mark.cli
@pytest.mark.essential
def test_validate_norm_layer_missing_mean_and_std(mock_config):
    mock_config.norm_config.add_norm_layer = True
    mock_config.norm_config.norm_mean = None
    mock_config.norm_config.norm_std = None
    with pytest.raises(ValueError, match=CLIErrorMessages.TRAINER.value.NORM_LAYER_MISSING_MEAN_OR_STD.value):
        _validate_norm_layer(mock_config)
