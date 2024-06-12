import torch
from unittest.mock import MagicMock, patch
import pytest

from advsecurenet.attacks.decision_based.boundary import DecisionBoundary
from advsecurenet.shared.types.configs import attack_configs


dummy_config = attack_configs.DecisionBoundaryAttackConfig()
dummy_model = MagicMock()
dummy_images = torch.rand(1, 3, 28, 28)
dummy_labels = torch.randint(0, 10, (1,))


@pytest.fixture
def attack_instance():
    return DecisionBoundary(config=dummy_config)

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_initialization(attack_instance):
    assert attack_instance.initial_delta == dummy_config.initial_delta
    assert attack_instance.initial_epsilon == dummy_config.initial_epsilon
    assert attack_instance.max_delta_trials == dummy_config.max_delta_trials
    assert attack_instance.max_epsilon_trials == dummy_config.max_epsilon_trials
    assert attack_instance.max_iterations == dummy_config.max_iterations
    assert attack_instance.max_initialization_trials == dummy_config.max_initialization_trials
    assert attack_instance.step_adapt == dummy_config.step_adapt
    assert attack_instance.verbose == dummy_config.verbose
    assert attack_instance.early_stopping == dummy_config.early_stopping
    assert attack_instance.early_stopping_threshold == dummy_config.early_stopping_threshold
    assert attack_instance.early_stopping_patience == dummy_config.early_stopping_patience

@pytest.mark.advsecurenet
@pytest.mark.comprehensive
def test_attack(attack_instance):
    # Mock necessary methods and attributes
    attack_instance._initialize = MagicMock(return_value=dummy_images)
    attack_instance._perturb_orthogonal = MagicMock(return_value=(dummy_images, dummy_config.initial_delta))
    attack_instance._perturb_forward = MagicMock(return_value=(dummy_images, dummy_config.initial_epsilon))
    attack_instance._update_best_images = MagicMock(return_value=(dummy_images, torch.tensor([0.5])))

    # Call the attack method
    adversarial_images = attack_instance.attack(dummy_model, dummy_images, dummy_labels)

    # Assertions
    assert adversarial_images.shape == dummy_images.shape
    assert isinstance(adversarial_images, torch.Tensor)

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_orthogonal_perturb(attack_instance):
    delta = 0.1
    perturbed = attack_instance._orthogonal_perturb(delta, dummy_images, dummy_images)
    assert perturbed.shape == dummy_images.shape
    assert isinstance(perturbed, torch.Tensor)

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_forward_perturb(attack_instance):
    epsilon = 0.1
    perturbed = attack_instance._forward_perturb(epsilon, dummy_images, dummy_images)
    assert perturbed.shape == dummy_images.shape
    assert isinstance(perturbed, torch.Tensor)