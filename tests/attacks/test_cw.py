import torch
import pytest
import warnings

from torch import nn
from pytest_mock import mocker
from advsecurenet.attacks import CWAttack
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types.configs.attack_configs import CWAttackConfig
from advsecurenet.shared.types.device import DeviceType

warnings.simplefilter("always", category=ImportWarning)


@pytest.fixture
def setup():
    # Define the model
    model = ModelFactory.get_model(
        "resnet18", pretrained=True, num_classes=1000)
    model.eval()

    # Define the input image and label, input image
    x = torch.rand((1, 3, 10, 10))
    y = torch.tensor([1])

    max_iterations = 1
    binary_search_steps = 1

    device = DeviceType.CPU
    config = CWAttackConfig(max_iterations=max_iterations,
                            binary_search_steps=binary_search_steps, device=device)

    return model, x, y, config


def create_cnn_model(channels, device):
    model = nn.Sequential(
        nn.Conv2d(channels, 8, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8*10*10, 2),
        nn.Softmax(dim=1)
    ).to(device)

    return model


def fake_forward_pass(*args, **kwargs):
    # This fake function always returns the true label, regardless of the input - this is to test the untargeted attack
    batch_size = args[0].shape[0]
    # Assuming 1000 classes, and the correct label is class "1"
    correct_labels = torch.ones((batch_size, 1000))
    # Assigning high score to class "1" so it's always chosen
    correct_labels[:, 1] = 10
    return correct_labels


def test_untargeted_attack(mocker, setup):
    model, x, y, config = setup

    # Mock the model's forward pass to always return the true label
    mocker.patch.object(model, 'forward', side_effect=fake_forward_pass)

    attack = CWAttack(config)

    # Generate the adversarial example
    adv_x = attack.attack(model, x, y)

    # Check that the adversarial example is different from the original image
    assert torch.all(adv_x != x)


def test_targeted_attack(setup):
    model, x, y, config = setup

    # set targeted to True
    config.targeted = True

    # Define the attack
    attack = CWAttack(config)

    # Generate the adversarial example
    adv_x = attack.attack(model, x, y)

    # Check that the adversarial example is different from the original image
    assert torch.all(adv_x != x)


def test_clip_min_max(setup):
    model, x, y, config = setup

    # set clip_min and clip_max to 0 and 1 respectively
    config.clip_min = 0
    config.clip_max = 1

    # Define the attack
    attack = CWAttack(config)

    # Generate the adversarial example
    adv_x = attack.attack(model, x, y)

    # Check that the attack was successful
    assert torch.argmax(model(adv_x)).item() != y.item()

    # Check that the pixel values are within the specified range
    assert torch.all(adv_x >= 0)
    assert torch.all(adv_x <= 1)


def test_device(setup):
    model, x, y, config = setup

    config.device = DeviceType.CPU

    # Define the attack
    attack = CWAttack(config)

    # Generate the adversarial example
    adv_x = attack.attack(model, x, y)

    # Check that the attack was successful
    assert torch.argmax(model(adv_x)).item() != y.item()
