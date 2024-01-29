"""
Unit tests for the attack success rate metric.
"""

from unittest.mock import MagicMock

import pytest
import torch

from advsecurenet.evaluation.evaluators.attack_success_rate_evaluator import \
    AttackSuccessRateEvaluator


@pytest.fixture
def mock_model():
    model = MagicMock()
    # Providing a fixed output for the mock model
    fixed_output = torch.tensor([[0.1] * 10] * 10)
    model.return_value = fixed_output
    return model


@pytest.fixture
def evaluator():
    return AttackSuccessRateEvaluator()


def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test_initialization(evaluator):
    assert evaluator.total_successful_attacks == 0
    assert evaluator.total_samples == 0


def test_reset(evaluator):
    evaluator.total_successful_attacks = 10
    evaluator.total_samples = 20
    evaluator.reset()
    assert evaluator.total_successful_attacks == 0
    assert evaluator.total_samples == 0


def test_update_non_targeted(mock_model, evaluator):
    set_random_seeds()  # Setting the random seed
    original_images = torch.rand(10, 3, 224, 224)
    true_labels = torch.randint(0, 10, (10,))
    adversarial_images = torch.rand(10, 3, 224, 224)

    evaluator.update(mock_model, original_images,
                     true_labels, adversarial_images)

    # assert evaluator.total_successful_attacks == 9
    assert evaluator.total_samples == 1


def test_update_targeted(mock_model, evaluator):
    set_random_seeds()  # Setting the random seed
    original_images = torch.rand(10, 3, 224, 224)
    true_labels = torch.randint(0, 10, (10,))
    adversarial_images = torch.rand(10, 3, 224, 224)
    target_labels = torch.randint(0, 10, (10,))

    evaluator.update(mock_model, original_images, true_labels,
                     adversarial_images, is_targeted=True, target_labels=target_labels)

    # assert evaluator.total_successful_attacks == 1
    assert evaluator.total_samples == 1


def test_update_targeted_without_target_labels(mock_model, evaluator):
    original_images = torch.rand(10, 3, 224, 224)
    true_labels = torch.randint(0, 10, (10,))
    adversarial_images = torch.rand(10, 3, 224, 224)

    with pytest.raises(ValueError):
        evaluator.update(mock_model, original_images, true_labels,
                         adversarial_images, is_targeted=True)


def test_get_results(evaluator):
    evaluator.total_successful_attacks = 5
    evaluator.total_samples = 10
    assert evaluator.get_results() == 0.5

    evaluator.reset()
    assert evaluator.get_results() == 0.0
