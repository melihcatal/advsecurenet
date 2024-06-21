# test_fgsm.py

from unittest.mock import Mock

import pytest
import torch

from advsecurenet.attacks.gradient_based.fgsm import FGSM
from advsecurenet.models.base_model import BaseModel
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types.configs.attack_configs.fgsm_attack_config import \
    FgsmAttackConfig
from advsecurenet.shared.types.configs.device_config import DeviceConfig
from advsecurenet.shared.types.configs.model_config import CreateModelConfig


@pytest.fixture
def device(request):
    device_arg = request.config.getoption("--device")
    return torch.device(device_arg if device_arg else "cpu")


@pytest.fixture
def mock_config(device):
    device_cfg = DeviceConfig(
        processor=device,
    )
    config = FgsmAttackConfig(
        epsilon=0.3,
        device=device_cfg,
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
    model.return_value = torch.zeros((1, 10), device=device)
    return model


@pytest.fixture
def mock_tensors(device):
    x = torch.randn((1, 3, 32, 32), device=device, dtype=torch.float32)
    y = torch.randint(0, 10, (1,), device=device, dtype=torch.long)
    return x, y


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_fgsm_init(mock_config):
    attack = FGSM(mock_config)
    assert attack.epsilon == mock_config.epsilon


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
def test_attack(mock_config, mock_model, mock_tensors):
    x, y = mock_tensors
    attack = FGSM(mock_config)
    adv_x = attack.attack(mock_model, x, y)
    assert isinstance(adv_x, torch.Tensor)
    assert adv_x.shape == x.shape


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
def test_fgsm_attack(mock_config, mock_tensors):
    x, _ = mock_tensors
    data_grad = torch.randn_like(x)
    attack = FGSM(mock_config)
    perturbed_image = attack._fgsm_attack(x, data_grad)
    assert isinstance(perturbed_image, torch.Tensor)
    assert perturbed_image.shape == x.shape


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
def test_adversarial_example_different(mock_config, mock_model, mock_tensors):
    x, y = mock_tensors
    attack = FGSM(mock_config)
    adv_x = attack.attack(mock_model, x, y)
    # Ensure the adversarial example is different from the original input
    assert not torch.equal(x, adv_x)


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
def test_targeted_attack(mock_config, mock_model, mock_tensors):
    mock_config.targeted = True
    x, y = mock_tensors
    attack = FGSM(mock_config)
    adv_x = attack.attack(mock_model, x, y)
    assert isinstance(adv_x, torch.Tensor)
    assert adv_x.shape == x.shape
    # Ensure the adversarial example is different from the original input
    assert not torch.equal(x, adv_x)
