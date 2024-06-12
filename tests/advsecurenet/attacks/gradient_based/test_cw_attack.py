# test_cw_attack.py

from unittest.mock import patch

import pytest
import torch

from advsecurenet.attacks.gradient_based.cw import CWAttack
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types.configs.attack_configs import CWAttackConfig
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
    config = CWAttackConfig(
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
def test_cw_attack_init(mock_config):
    attack = CWAttack(mock_config)
    assert attack.c_init == mock_config.c_init
    assert attack.kappa == mock_config.kappa
    assert attack.learning_rate == mock_config.learning_rate
    assert attack.max_iterations == mock_config.max_iterations
    assert attack.abort_early == mock_config.abort_early
    assert attack.targeted == mock_config.targeted
    assert attack.binary_search_steps == mock_config.binary_search_steps
    assert attack.clip_min == mock_config.clip_min
    assert attack.clip_max == mock_config.clip_max
    assert attack.c_lower == mock_config.c_lower
    assert attack.c_upper == mock_config.c_upper
    assert attack.patience == mock_config.patience


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
@patch('advsecurenet.attacks.gradient_based.cw.CWAttack._run_attack', return_value=torch.randn((1, 3, 32, 32), dtype=torch.float32))
@patch('advsecurenet.attacks.gradient_based.cw.CWAttack._is_successful', return_value=torch.tensor([True], dtype=torch.bool))
def test_attack(mock_run_attack, mock_is_successful, mock_config, mock_model, mock_tensors):
    x, y = mock_tensors

    # Move the mock return values to the device of x
    mock_run_attack.return_value = mock_run_attack.return_value.to(x.device)
    mock_is_successful.return_value = mock_is_successful.return_value.to(
        x.device)

    attack = CWAttack(mock_config)
    adv_x = attack.attack(mock_model, x, y)

    assert isinstance(adv_x, torch.Tensor)
    assert adv_x.shape == x.shape
    mock_run_attack.assert_called()
    mock_is_successful.assert_called()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_initialize_x(mock_config, mock_tensors):
    x, _ = mock_tensors
    attack = CWAttack(mock_config)
    x_initialized = attack._initialize_x(x)
    assert isinstance(x_initialized, torch.Tensor)
    assert x_initialized.shape == x.shape


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
def test_run_attack(mock_config, mock_model, mock_tensors):
    x, y = mock_tensors
    attack = CWAttack(mock_config)
    adv_x = attack._run_attack(mock_model, x, y)
    assert isinstance(adv_x, torch.Tensor)
    assert adv_x.shape == x.shape


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_cw_loss(mock_config, mock_model, mock_tensors):
    x, y = mock_tensors
    perturbation = torch.zeros_like(x, requires_grad=True, device=x.device)
    attack = CWAttack(mock_config)
    loss = attack._cw_loss(mock_model, x, y, perturbation)
    assert isinstance(loss, torch.Tensor)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_is_successful(mock_config, mock_model, mock_tensors):
    x, y = mock_tensors
    attack = CWAttack(mock_config)
    is_successful = attack._is_successful(mock_model, x, y)
    assert isinstance(is_successful, torch.Tensor)
    assert is_successful.dtype == torch.bool
