import pytest
import torch
from torch import nn

from advsecurenet.attacks import DeepFool
from advsecurenet.shared.types.configs.attack_configs import \
    DeepFoolAttackConfig
from advsecurenet.shared.types.configs.device_config import DeviceConfig


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        return self.fc(x.view(-1, 28*28))


@pytest.fixture
def device(request):
    device_arg = request.config.getoption("--device")
    return torch.device(device_arg if device_arg else "cpu")


@pytest.fixture
def mock_model(device):
    model = SimpleModel()
    model.eval()
    model = model.to(device)
    model.return_value = torch.zeros((1, 10), device=device)
    return model


@pytest.fixture
def setup_attack(device):
    device_cfg = DeviceConfig(
        processor=device,
    )
    config = DeepFoolAttackConfig(device=device_cfg)
    return config


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_deepfool_initialization(setup_attack):
    attack = DeepFool(setup_attack)
    assert attack.num_classes == 10
    assert attack.overshoot == 0.02
    assert attack.max_iterations == 50


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_deepfool_device_configuration_cpu():
    device_cfg = DeviceConfig(
        processor=torch.device("cpu"),
    )
    config = DeepFoolAttackConfig(device=device_cfg)
    attack = DeepFool(config)
    assert attack.device_manager.initial_device == torch.device("cpu")


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
def test_deepfool_attack_outcome(mock_model, setup_attack):
    attack = DeepFool(setup_attack)
    sample_image = torch.rand(
        (1, 1, 28, 28), requires_grad=True, device=attack.device_manager.initial_device)
    sample_label = torch.tensor(
        [7], device=attack.device_manager.initial_device)
    x_adv = attack.attack(mock_model, sample_image, sample_label)
    assert not torch.equal(x_adv, sample_image)


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
def test_deepfool_attack_termination(mock_model, setup_attack):
    setup_attack.max_iterations = 5
    attack = DeepFool(setup_attack)
    sample_image = torch.rand(
        (1, 1, 28, 28), requires_grad=True, device=attack.device_manager.initial_device)
    sample_label = torch.tensor(
        [7], device=attack.device_manager.initial_device)
    x_adv = attack.attack(mock_model, sample_image, sample_label)
    assert (x_adv - sample_image).abs().sum() > 0


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
def test_deepfool_batch_attack(mock_model, setup_attack):
    attack = DeepFool(setup_attack)
    x = torch.rand((2, 1, 28, 28), requires_grad=True,
                   device=attack.device_manager.initial_device)
    y = torch.tensor([7, 2], device=attack.device_manager.initial_device)
    x_adv = attack.attack(mock_model, x, y)
    assert not torch.equal(x_adv, x)
