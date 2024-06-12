from unittest.mock import MagicMock, patch

import pytest
import torch

from advsecurenet.evaluation.evaluators.robustness_gap_evaluator import \
    RobustnessGapEvaluator
from advsecurenet.models.base_model import BaseModel


@pytest.fixture
def evaluator():
    return RobustnessGapEvaluator()


@pytest.fixture
def mock_model():
    model = MagicMock(spec=BaseModel)
    return model


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_initialization(evaluator):
    assert evaluator.total_correct_clean == 0
    assert evaluator.total_correct_adv == 0
    assert evaluator.total_samples == 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_reset(evaluator):
    evaluator.total_correct_clean = 10
    evaluator.total_correct_adv = 8
    evaluator.total_samples = 12
    evaluator.reset()
    assert evaluator.total_correct_clean == 0
    assert evaluator.total_correct_adv == 0
    assert evaluator.total_samples == 0


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.evaluation.evaluators.robustness_gap_evaluator.RobustnessGapEvaluator._calculate_acc', return_value=(5, 3))
def test_update(mock_calculate_acc, evaluator, mock_model):
    original_images = torch.zeros((10, 3, 32, 32))
    true_labels = torch.tensor([1] * 10)
    adversarial_images = torch.ones((10, 3, 32, 32))

    evaluator.update(mock_model, original_images,
                     true_labels, adversarial_images)

    assert evaluator.total_correct_clean == 5
    assert evaluator.total_correct_adv == 3
    assert evaluator.total_samples == 10


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_results(evaluator):
    evaluator.total_correct_clean = 5
    evaluator.total_correct_adv = 3
    evaluator.total_samples = 10

    results = evaluator.get_results()

    assert results["clean_accuracy"] == 0.5
    assert results["adversarial_accuracy"] == 0.3
    assert results["robustness_gap"] == 0.2

    evaluator.reset()
    results = evaluator.get_results()
    assert results["clean_accuracy"] == 0.0
    assert results["adversarial_accuracy"] == 0.0
    assert results["robustness_gap"] == 0.0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_calculate_acc(evaluator, mock_model):
    clean_images = torch.randn((10, 3, 32, 32))
    adv_images = torch.randn((10, 3, 32, 32))
    labels = torch.randint(0, 10, (10,))

    mock_model.return_value = torch.randn((10, 10))  # Simulating logits output

    clean_correct, adv_correct = evaluator._calculate_acc(
        mock_model, clean_images, adv_images, labels)

    assert isinstance(clean_correct, int)
    assert isinstance(adv_correct, int)
