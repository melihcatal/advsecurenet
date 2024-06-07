from unittest.mock import MagicMock, patch

import pytest
import torch

from advsecurenet.attacks.decision_based.boundary import DecisionBoundary
from advsecurenet.shared.types.configs import attack_configs

# Assume there are existing mock or dummy objects
dummy_config = attack_configs.DecisionBoundaryAttackConfig()
dummy_model = MagicMock()
dummy_images = torch.rand(1, 3, 28, 28)  # Example shape for an image
dummy_labels = torch.randint(0, 10, (1,))  # Example labels


@pytest.fixture
def attack_instance():
    return DecisionBoundary(config=dummy_config)


def test_initialization(attack_instance):
    assert attack_instance.initial_delta == dummy_config.initial_delta
    # Add assertions for all the other configuration parameters


def test_orthogonal_perturb(attack_instance):
    delta = 0.1  # Example value for delta
    perturbed = attack_instance._orthogonal_perturb(
        delta, dummy_images, dummy_images)
    assert perturbed.shape == dummy_images.shape  # Ensure the shape is correct
    # Add other assertions as necessary


def test_forward_perturb(attack_instance):
    epsilon = 0.1  # Example value for epsilon
    perturbed = attack_instance._forward_perturb(
        epsilon, dummy_images, dummy_images)
    assert perturbed.shape == dummy_images.shape  # Ensure the shape is correct
    # Add other assertions as necessary
