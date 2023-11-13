import torch
import pytest
from advsecurenet.attacks import FGSM
from advsecurenet.shared.types.configs.attack_configs import FgsmAttackConfig


@pytest.fixture(scope="module")
def setup_attack():
    device = torch.device("cpu")
    epsilon = 0.3
    config = FgsmAttackConfig(epsilon=epsilon, device=device)
    attack = FGSM(config)
    return device, epsilon, attack


def create_simple_model(device, input_size=10):
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
        torch.nn.Softmax(dim=1)
    ).to(device)
    return model


def create_cnn_model(device, channels):
    model = torch.nn.Sequential(
        torch.nn.Conv2d(channels, 8, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(8*10*10, 2),
        torch.nn.Softmax(dim=1)
    ).to(device)
    return model


def test_attack(setup_attack):
    device, _, attack = setup_attack
    model = create_simple_model(device)
    x = torch.randn(1, 10).to(device)
    y = torch.tensor([0], dtype=torch.long).to(device)
    perturbed_image = attack.attack(model, x, y)
    assert not torch.all(torch.eq(perturbed_image, x))


def test_epsilon(setup_attack):
    _, epsilon, attack = setup_attack
    assert attack.epsilon == epsilon


def test_device(setup_attack):
    device, _, attack = setup_attack
    assert attack.device == device
