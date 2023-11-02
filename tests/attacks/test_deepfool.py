import torch
import pytest
from torch import nn
from advsecurenet.attacks import DeepFool
from advsecurenet.shared.types import  DeviceType
from advsecurenet.shared.types.configs.attack_configs import DeepFoolAttackConfig


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        return self.fc(x.view(-1, 28*28))


@pytest.fixture(scope="module")
def setup_attack():
    device = DeviceType.CPU
    config = DeepFoolAttackConfig(device=device)
    return config


def test_deepfool_initialization(setup_attack):
    attack = DeepFool(setup_attack)
    assert attack.num_classes == 10
    assert attack.overshoot == 0.02
    assert attack.max_iterations == 50


def test_deepfool_device_configuration(setup_attack):
    # default is CPU
    attack = DeepFool(setup_attack)
    assert attack.device == DeviceType.CPU.value


def test_deepfool_attack_outcome(setup_attack):
    model = SimpleModel().eval()
    attack = DeepFool(setup_attack)
    sample_image = torch.rand((1, 1, 28, 28))
    sample_label = torch.tensor([7])
    x_adv = attack.attack(model, sample_image, sample_label)
    assert not torch.equal(x_adv, sample_image)


def test_deepfool_attack_termination(setup_attack):
    model = SimpleModel().eval()
    setup_attack.max_iterations = 5
    attack = DeepFool(setup_attack)
    sample_image = torch.rand((1, 1, 28, 28))
    sample_label = torch.tensor([7])
    x_adv = attack.attack(model, sample_image, sample_label)
    assert (x_adv - sample_image).abs().sum() > 0


def test_deepfool_batch_attack(setup_attack):
    model = SimpleModel().eval()
    attack = DeepFool(setup_attack)
    x = torch.rand((2, 1, 28, 28))
    y = torch.tensor([7, 2])
    x_adv = attack.attack(model, x, y)
    assert not torch.equal(x_adv, x)
