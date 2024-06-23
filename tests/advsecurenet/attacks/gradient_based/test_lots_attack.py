from unittest.mock import patch

import pytest
import torch
from torch import optim

from advsecurenet.attacks.gradient_based.lots import LOTS
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


@pytest.fixture
def mock_attack(mock_config):
    return LOTS(mock_config)


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
@patch('advsecurenet.attacks.gradient_based.lots.LOTS._lots_iterative', return_value=torch.randn((1, 3, 32, 32)))
def test_attack_iterative(mock_lots_iterative, mock_config, mock_model, mock_tensors):
    mock_config.mode = LotsAttackMode.ITERATIVE
    x, y, x_target = mock_tensors
    attack = LOTS(mock_config)
    adv_x = attack.attack(mock_model, x, y, x_target)
    assert isinstance(adv_x, torch.Tensor)
    assert adv_x.shape == x.shape
    mock_lots_iterative.assert_called()


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
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
@patch('advsecurenet.attacks.gradient_based.lots.LOTS._lots_single', return_value=torch.randn((1, 3, 32, 32)))
@patch('advsecurenet.attacks.gradient_based.lots.LOTS._validate_config', return_value=None)
def test_attack_invalid(mock_validate_config, mock_lots_single, mock_config, mock_model, mock_tensors):
    mock_config.mode = "invalid_mode"
    x, y, x_target = mock_tensors
    with pytest.raises(ValueError, match="Invalid mode provided."):
        attack = LOTS(mock_config)
        attack.attack(mock_model, x, y, x_target)


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
@patch('advsecurenet.attacks.gradient_based.lots.LOTS._lots_single', return_value=torch.randn((1, 3, 32, 32)))
def test_attack_single_distributed(mock_lots_single, mock_config, mock_model, mock_tensors):

    mock_config.mode = LotsAttackMode.SINGLE
    x, y, x_target = mock_tensors
    attack = LOTS(mock_config)
    adv_x = attack.attack(mock_model, x, y, x_target)

    assert isinstance(adv_x, torch.Tensor)
    assert adv_x.shape == x.shape
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


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adversarial_success_with_target_classes(mock_config, mock_model, mock_attack):
    x = torch.randn(5, 3, 32, 32, device=mock_config.device.processor)
    x_deep_features = torch.randn(5, 100, device=mock_config.device.processor)
    x_target_deep_features = torch.randn(
        5, 100, device=mock_config.device.processor)
    y = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long,
                     device=mock_config.device.processor)

    success = mock_attack._evaluate_adversarial_success(
        mock_model, x, x_deep_features, x_target_deep_features, y)

    assert success.shape == (5,)
    assert success.dtype == torch.bool


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adversarial_success_without_target_classes(mock_config, mock_model, mock_attack):
    x = torch.randn(5, 3, 32, 32, device=mock_config.device.processor)
    x_deep_features = torch.randn(5, 100, device=mock_config.device.processor)
    x_target_deep_features = torch.randn(
        5, 100, device=mock_config.device.processor)

    success = mock_attack._evaluate_adversarial_success(
        mock_model, x, x_deep_features, x_target_deep_features)

    assert success.shape == (5,)
    assert success.dtype == torch.bool


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adversarial_success_within_epsilon(mock_config, mock_model, mock_attack):
    x = torch.randn(5, 3, 32, 32, device=mock_config.device.processor)
    x_deep_features = torch.randn(5, 100, device=mock_config.device.processor)
    x_target_deep_features = x_deep_features + 0.1
    distances = torch.norm(x_deep_features - x_target_deep_features, dim=1)
    mock_attack._epsilon = distances.max().item() + 0.1

    success = mock_attack._evaluate_adversarial_success(
        mock_model, x, x_deep_features, x_target_deep_features)

    assert success.all()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adversarial_failure_outside_epsilon(mock_config, mock_model, mock_attack):
    x = torch.randn(5, 3, 32, 32, device=mock_config.device.processor)
    x_deep_features = torch.randn(5, 100, device=mock_config.device.processor)
    x_target_deep_features = x_deep_features + 1.0

    success = mock_attack._evaluate_adversarial_success(
        mock_model, x, x_deep_features, x_target_deep_features)

    assert not success.any()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adversarial_success_combined_criteria(mock_config, mock_model, mock_attack):
    x = torch.randn(5, 3, 32, 32, device=mock_config.device.processor)
    x_deep_features = torch.randn(5, 100, device=mock_config.device.processor)
    x_target_deep_features = x_deep_features + 0.1
    y = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long,
                     device=mock_config.device.processor)
    distances = torch.norm(x_deep_features - x_target_deep_features, dim=1)
    mock_attack._epsilon = distances.max().item() + 0.1

    success = mock_attack._evaluate_adversarial_success(
        mock_model, x, x_deep_features, x_target_deep_features, y)

    assert success.shape == (5,)
    assert success.dtype == torch.bool
    assert success.all()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_validate_config_valid(mock_attack):
    config = LotsAttackConfig(mode=LotsAttackMode.SINGLE,
                              epsilon=0.1,
                              learning_rate=0.01,
                              max_iterations=10,
                              deep_feature_layer='layer3')
    mock_attack._validate_config(config)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_validate_config_invalid_type(mock_attack):
    config = "invalid_config"
    with pytest.raises(ValueError, match="Invalid config type provided. Expected LotsAttackConfig. But got: <class 'str'>"):
        mock_attack._validate_config(config)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_validate_config_invalid_mode(mock_config, mock_attack):
    mock_config.mode = "invalid_mode"

    with pytest.raises(ValueError, match="Invalid mode type provided. Allowed modes are: iterative, single"):
        mock_attack._validate_config(mock_config)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_validate_config_negative_epsilon(mock_config, mock_attack):
    mock_config.epsilon = -0.1
    with pytest.raises(ValueError, match="Invalid epsilon value provided. Epsilon must be greater than 0."):
        mock_attack._validate_config(mock_config)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_validate_config_negative_learning_rate(mock_config, mock_attack):
    mock_config.learning_rate = -0.1
    with pytest.raises(ValueError, match="Invalid learning rate value provided. Learning rate must be greater than 0."):
        mock_attack._validate_config(mock_config)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_validate_config_negative_max_iterations(mock_config, mock_attack):
    mock_config.max_iterations = -10
    with pytest.raises(ValueError, match="Invalid max iterations value provided. Max iterations must be greater than 0."):
        mock_attack._validate_config(mock_config)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_validate_config_missing_deep_feature_layer(mock_config, mock_attack):
    mock_config.deep_feature_layer = None
    with pytest.raises(ValueError, match="Deep feature layer that you want to use for the attack must be provided."):
        mock_attack._validate_config(mock_config)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_lots_iterative_success(mock_config, mock_attack, mock_model):
    mock_config.mode = LotsAttackMode.ITERATIVE
    mock_config.max_iterations = 2

    attack = LOTS(mock_config)
    feature_extractor_model = attack._create_feature_extractor(mock_model)
    x = torch.randn(5, 3, 32, 32,  device=mock_config.device.processor)
    x_target = torch.randn(5, 3, 32, 32, device=mock_config.device.processor)
    y = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long,
                     device=mock_config.device.processor)
    optimizer = optim.Adam([x], lr=mock_config.learning_rate)

    perturbed_x = attack._lots_iterative(
        mock_model, x, x_target, feature_extractor_model, optimizer, y)

    assert perturbed_x.shape == x.shape
    assert perturbed_x.min() >= 0
    assert perturbed_x.max() <= 1


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_lots_iterative_no_target_classes(mock_config, mock_attack, mock_model):
    mock_config.mode = LotsAttackMode.ITERATIVE
    mock_config.max_iterations = 2

    attack = LOTS(mock_config)
    feature_extractor_model = attack._create_feature_extractor(mock_model)

    x = torch.randn(5, 3, 32, 32,  device=mock_config.device.processor)
    x_target = torch.randn(5, 3, 32, 32, device=mock_config.device.processor)
    optimizer = optim.Adam([x], lr=mock_config.learning_rate)

    perturbed_x = attack._lots_iterative(
        mock_model, x, x_target, feature_extractor_model, optimizer)

    assert perturbed_x.shape == x.shape
    assert perturbed_x.min() >= 0
    assert perturbed_x.max() <= 1


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_lots_iterative_early_success(mock_config, mock_attack, mock_model):

    mock_config.mode = LotsAttackMode.ITERATIVE
    mock_config.max_iterations = 2

    attack = LOTS(mock_config)
    feature_extractor_model = attack._create_feature_extractor(mock_model)

    x = torch.randn(5, 3, 32, 32, device=mock_config.device.processor)
    x_target = torch.randn(5, 3, 32, 32, device=mock_config.device.processor)
    y = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long,
                     device=mock_config.device.processor)
    optimizer = optim.Adam([x], lr=mock_config.learning_rate)

    def mock_evaluate_adversarial_success(model, x, x_deep_features, x_target_deep_feature, y):
        success = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        success[0] = True
        return success

    attack._evaluate_adversarial_success = mock_evaluate_adversarial_success

    perturbed_x = attack._lots_iterative(
        mock_model, x, x_target, feature_extractor_model, optimizer, y)

    assert perturbed_x.shape == x.shape
    assert perturbed_x.min() >= 0
    assert perturbed_x.max() <= 1


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_validate_layer_exists(mock_model, mock_config):
    layer_name = "model.fc2"
    mock_config.deep_feature_layer = layer_name
    attack = LOTS(mock_config)
    attack._validate_layer(mock_model)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_validate_layer_not_exists(mock_model, mock_config):
    layer_name = "model.NOT_EXISTING_LAYER"
    mock_config.deep_feature_layer = layer_name
    with pytest.raises(ValueError) as excinfo:
        attack = LOTS(mock_config)
        attack._validate_layer(mock_model)
    assert f"Layer '{layer_name}' not found in the model." in str(
        excinfo.value)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_validate_layer_not_starts_wth_model(mock_model, mock_config):
    layer_name = "fc2"
    mock_config.deep_feature_layer = layer_name
    attack = LOTS(mock_config)
    attack._validate_layer(mock_model)
    assert attack._deep_feature_layer == f"model.{layer_name}"
