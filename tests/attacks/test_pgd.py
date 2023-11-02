import torch
import torch.nn as nn
import pytest
from advsecurenet.attacks import PGD
from advsecurenet.shared.types import DeviceType
from advsecurenet.shared.types.configs.attack_configs import PgdAttackConfig

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        return self.fc(x.view(-1, 28*28))


@pytest.fixture
def pgd_attack():
    return PGD(PgdAttackConfig)


@pytest.fixture
def simple_model():
    return SimpleModel().eval()


@pytest.fixture
def sample_image():
    return torch.rand((1, 1, 28, 28))


@pytest.fixture
def sample_label():
    return torch.tensor([7])


def test_initialization(pgd_attack):
    assert pgd_attack.epsilon == PgdAttackConfig.epsilon
    assert pgd_attack.alpha == PgdAttackConfig.alpha
    assert pgd_attack.num_iter == PgdAttackConfig.num_iter


def test_device_configuration(pgd_attack):
    assert pgd_attack.device == DeviceType.CPU.value


def test_attack_outcome(pgd_attack, simple_model, sample_image, sample_label):
    adv_x = pgd_attack.attack(simple_model, sample_image, sample_label)
    assert not torch.equal(adv_x, sample_image)


def test_clamping(pgd_attack, simple_model, sample_image, sample_label):
    adv_x = pgd_attack.attack(simple_model, sample_image, sample_label)
    assert (adv_x >= 0).all()
    assert (adv_x <= 1).all()


def test_targeted_untargeted(pgd_attack, simple_model, sample_image, sample_label):
    #  Ensure the targeted attack reduces the loss and the untargeted attack increases it.
    target_label = torch.tensor([2])  # A different label
    adv_x_targeted = pgd_attack.attack(
        simple_model, sample_image, target_label, targeted=True)
    adv_x_untargeted = pgd_attack.attack(
        simple_model, sample_image, sample_label, targeted=False)

    outputs_targeted = simple_model(adv_x_targeted)
    outputs_untargeted = simple_model(adv_x_untargeted)

    loss_targeted = torch.nn.functional.cross_entropy(
        outputs_targeted, target_label)
    loss_untargeted = torch.nn.functional.cross_entropy(
        outputs_untargeted, sample_label)

    # For a targeted attack, the loss should be lower
    assert loss_targeted.item() < loss_untargeted.item()
