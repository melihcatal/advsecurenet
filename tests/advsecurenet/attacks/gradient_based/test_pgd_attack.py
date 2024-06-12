import pytest
import torch
import torch.nn as nn

from advsecurenet.attacks import PGD
from advsecurenet.shared.types.configs.attack_configs import PgdAttackConfig
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
def pgd_attack(device):
    device_cfg = DeviceConfig(
        processor=device,
    )
    cfg = PgdAttackConfig(device=device_cfg)
    return PGD(config=cfg)


@pytest.fixture
def simple_model(device):
    model = SimpleModel()
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def sample_image(device):
    return torch.rand((1, 1, 28, 28), device=device, dtype=torch.float32)


@pytest.fixture
def sample_label(device):
    return torch.tensor([7], device=device, dtype=torch.long)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_initialization(pgd_attack):
    assert pgd_attack.epsilon == PgdAttackConfig.epsilon
    assert pgd_attack.alpha == PgdAttackConfig.alpha
    assert pgd_attack.num_iter == PgdAttackConfig.num_iter


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_device_configuration_cpu():
    device_cfg = DeviceConfig(
        processor=torch.device("cpu"),
    )
    cfg = PgdAttackConfig(device=device_cfg)
    attack = PGD(config=cfg)
    assert attack.device_manager.initial_device == torch.device("cpu")


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_attack_outcome(pgd_attack, simple_model, sample_image, sample_label):
    adv_x = pgd_attack.attack(simple_model, sample_image, sample_label)
    assert not torch.equal(adv_x, sample_image)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_clamping(pgd_attack, simple_model, sample_image, sample_label):
    adv_x = pgd_attack.attack(simple_model, sample_image, sample_label)
    assert (adv_x >= 0).all()
    assert (adv_x <= 1).all()


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
def test_targeted_attack(pgd_attack, simple_model, sample_image, sample_label):
    target_label = torch.tensor(
        [2], device=pgd_attack.device_manager.initial_device)  # A different label
    pgd_attack.targeted = True
    adv_x_targeted = pgd_attack.attack(
        simple_model, sample_image, target_label)
    assert not torch.equal(adv_x_targeted, sample_image)


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
def test_untargeted_attack(pgd_attack, simple_model, sample_image, sample_label):
    pgd_attack.targeted = False
    adv_x_untargeted = pgd_attack.attack(
        simple_model, sample_image, sample_label)
    assert not torch.equal(adv_x_untargeted, sample_image)
