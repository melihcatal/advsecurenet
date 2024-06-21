# test_lots.py

from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from advsecurenet.attacks.gradient_based.lots import LOTS
from advsecurenet.models.base_model import BaseModel
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types.configs.attack_configs import (LotsAttackConfig,
                                                              LotsAttackMode)
from advsecurenet.shared.types.configs.device_config import DeviceConfig
from advsecurenet.shared.types.configs.model_config import CreateModelConfig


@pytest.fixture
def device(request):
    device_arg = request.config.getoption("--device")
    return torch.device(device_arg if device_arg else "cpu")


@pytest.fixture
def mock_config(device):
    device_config = DeviceConfig(
        processor=device,
        use_ddp=False
    )

    config = LotsAttackConfig(
        deep_feature_layer="model.fc2",
        device=device_config,
        mode=LotsAttackMode.SINGLE,
    )

    return config


@pytest.fixture
def mock_model(device):

    model = ModelFactory.create_model(
        CreateModelConfig(
            model_name="CustomCifar10Model",
            num_classes=10,
            num_input_channels=3,
            pretrained=False
        )
    )
    model = model.to(device)
    model.eval()
    model.return_value = torch.zeros((1, 10), device=device)
    return model


@pytest.fixture
def mock_tensors(device):
    x = torch.randn((1, 3, 32, 32),
                    requires_grad=True,
                    device=device)
    y = torch.randint(0, 10, (1,), device=device)
    x_target = torch.randn((1, 3, 32, 32), device=device)
    return x, y, x_target


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_lots_init(mock_config):
    attack = LOTS(mock_config)
    assert attack._deep_feature_layer == mock_config.deep_feature_layer
    assert attack._mode == mock_config.mode
    assert attack._epsilon == mock_config.epsilon
    assert attack._learning_rate == mock_config.learning_rate
    assert attack._max_iterations == mock_config.max_iterations
    assert attack._verbose == mock_config.verbose


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
# @patch('advsecurenet.attacks.gradient_based.lots.LOTS._create_feature_extractor', return_value=Mock())
@patch('advsecurenet.attacks.gradient_based.lots.LOTS._lots_iterative', return_value=torch.randn((1, 3, 32, 32)))
def test_attack_iterative(mock_lots_iterative, mock_config, mock_model, mock_tensors):
    mock_config.mode = LotsAttackMode.ITERATIVE
    x, y, x_target = mock_tensors
    attack = LOTS(mock_config)
    adv_x = attack.attack(mock_model, x, y, x_target)
    assert isinstance(adv_x, torch.Tensor)
    assert adv_x.shape == x.shape
    # mock_create_feature_extractor.assert_called()
    mock_lots_iterative.assert_called()


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
# @patch('advsecurenet.attacks.gradient_based.lots.LOTS._create_feature_extractor', return_value=Mock())
@patch('advsecurenet.attacks.gradient_based.lots.LOTS._lots_single', return_value=torch.randn((1, 3, 32, 32)))
def test_attack_single(mock_lots_single, mock_config, mock_model, mock_tensors):
    mock_config.mode = LotsAttackMode.SINGLE
    x, y, x_target = mock_tensors
    attack = LOTS(mock_config)
    adv_x = attack.attack(mock_model, x, y, x_target)
    assert isinstance(adv_x, torch.Tensor)
    assert adv_x.shape == x.shape
    # mock_create_feature_extractor.assert_called()
    mock_lots_single.assert_called()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_prepare_inputs(mock_config, mock_tensors):
    x, _, x_target = mock_tensors
    attack = LOTS(mock_config)
    x_prepared, x_target_prepared = attack._prepare_inputs(x, x_target)
    assert isinstance(x_prepared, torch.Tensor)
    assert x_prepared.shape == x.shape
    assert isinstance(x_target_prepared, torch.Tensor)
    assert x_target_prepared.shape == x_target.shape


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adversarial_example_different(mock_config, mock_model, mock_tensors):
    x, y, x_target = mock_tensors
    attack = LOTS(mock_config)
    adv_x = attack.attack(mock_model, x, y, x_target)
    # Ensure the adversarial example is different from the original input
    assert not torch.equal(x, adv_x)
