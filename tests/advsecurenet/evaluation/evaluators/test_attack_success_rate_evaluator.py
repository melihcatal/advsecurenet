from unittest.mock import MagicMock, patch

import pytest
import torch

from advsecurenet.evaluation.evaluators.attack_success_rate_evaluator import \
    AttackSuccessRateEvaluator
from advsecurenet.models.base_model import BaseModel


@pytest.fixture
def evaluator():
    return AttackSuccessRateEvaluator()


@pytest.fixture
def mock_model():
    model = MagicMock(spec=BaseModel)
    model.return_value = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    return model


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_reset(evaluator):
    evaluator.total_successful_attacks = 10
    evaluator.total_samples = 20
    evaluator.reset()
    assert evaluator.total_successful_attacks == 0
    assert evaluator.total_samples == 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_results(evaluator):
    evaluator.total_successful_attacks = 8
    evaluator.total_samples = 10
    result = evaluator.get_results()
    assert result == 0.8

    evaluator.total_successful_attacks = 0
    evaluator.total_samples = 0
    result = evaluator.get_results()
    assert result == 0.0


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.evaluation.evaluators.attack_success_rate_evaluator.AttackSuccessRateEvaluator._predict')
def test_update_untargeted_attack(mock_predict, evaluator, mock_model):
    # Set up the mock return values for _predict calls
    mock_predict.side_effect = [torch.tensor([1, 0]), torch.tensor([0, 0])]

    original_images = torch.zeros((2, 3, 32, 32))
    true_labels = torch.tensor([1, 0])
    adversarial_images = torch.ones((2, 3, 32, 32))

    evaluator.update(mock_model, original_images, true_labels,
                     adversarial_images, is_targeted=False)

    assert evaluator.total_samples == 2
    assert evaluator.total_successful_attacks == 1


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.evaluation.evaluators.attack_success_rate_evaluator.AttackSuccessRateEvaluator._predict')
def test_update_targeted_attack(mock_predict, evaluator, mock_model):
    # Set up the mock return values for _predict calls
    mock_predict.side_effect = [torch.tensor([1, 0]), torch.tensor([0, 1])]

    original_images = torch.zeros((2, 3, 32, 32))
    true_labels = torch.tensor([1, 0])
    adversarial_images = torch.ones((2, 3, 32, 32))
    target_labels = torch.tensor([0, 1])

    evaluator.update(mock_model,
                     original_images,
                     true_labels,
                     adversarial_images,
                     is_targeted=True,
                     target_labels=target_labels)

    assert evaluator.total_samples == 2
    assert evaluator.total_successful_attacks == 2


@ pytest.mark.advsecurenet
@ pytest.mark.essential
def test_predict(evaluator, mock_model):
    images = torch.zeros((2, 3, 32, 32))
    prediction_labels = evaluator._predict(mock_model, images)
    assert prediction_labels.tolist() == [1, 0]


@ pytest.mark.advsecurenet
@ pytest.mark.essential
def test_get_correct_predictions_mask(evaluator):
    prediction_labels = torch.tensor([1, 0, 2])
    true_labels = torch.tensor([1, 1, 2])
    mask = evaluator._get_correct_predictions_mask(
        prediction_labels, true_labels)
    assert mask.tolist() == [True, False, True]


@ pytest.mark.advsecurenet
@ pytest.mark.essential
def test_filter_correct_predictions(evaluator):
    mask = torch.tensor([True, False, True])
    true_labels = torch.tensor([1, 0, 2])
    adversarial_images = torch.zeros((3, 3, 32, 32))
    correct_true_labels, correct_adversarial_images = evaluator._filter_correct_predictions(
        mask, true_labels, adversarial_images)

    assert correct_true_labels.tolist() == [1, 2]
    assert correct_adversarial_images.shape == (2, 3, 32, 32)


@ pytest.mark.advsecurenet
@ pytest.mark.essential
def test_validate_targeted_attack(evaluator):
    with pytest.raises(ValueError):
        evaluator._validate_targeted_attack(None)


@ pytest.mark.advsecurenet
@ pytest.mark.essential
def test_calculate_success_untargeted(evaluator):
    adversarial_labels = torch.tensor([1, 0, 1])
    true_labels = torch.tensor([1, 0, 2])
    success = evaluator._calculate_success(
        adversarial_labels, true_labels, is_targeted=False)
    assert success.item() == 1


@ pytest.mark.advsecurenet
@ pytest.mark.essential
def test_calculate_success_targeted(evaluator):
    adversarial_labels = torch.tensor([1, 1, 1])
    target_labels = torch.tensor([1, 1, 0])
    success = evaluator._calculate_success(adversarial_labels, torch.tensor(
        []), is_targeted=True, target_labels=target_labels)
    assert success.item() == 2


@ pytest.mark.advsecurenet
@ pytest.mark.essential
def test_update_metrics(evaluator):
    evaluator._update_metrics(torch.tensor(5), 10)
    assert evaluator.total_successful_attacks == 5
    assert evaluator.total_samples == 10
