from unittest.mock import MagicMock

import pytest
import torch

from advsecurenet.evaluation.evaluators.transferability_evaluator import \
    TransferabilityEvaluator


class DummyModel():
    def __init__(self, model_name):
        super().__init__()
        self._model_name = model_name

    def forward(self, x):
        return torch.zeros(x.size(0), 10)  # Dummy output

    def parameters(self):
        return iter([torch.randn(1)])

    def eval(self):
        pass

    def __call__(self, x):
        return self.forward(x)


@pytest.fixture
def dummy_models():
    return [DummyModel(f"model_{i}") for i in range(3)]


@pytest.fixture
def evaluator(dummy_models):
    return TransferabilityEvaluator(dummy_models)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_init(evaluator, dummy_models):
    assert evaluator.target_models == dummy_models
    assert isinstance(evaluator.transferability_data, dict)
    assert evaluator.total_successful_on_source == 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_reset(evaluator):
    evaluator.reset()
    assert isinstance(evaluator.transferability_data, dict)
    assert evaluator.total_successful_on_source == 0
    assert all(['successful_transfer' in data and 'successful_on_source' in data
                for data in evaluator.transferability_data.values()])


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
def test_update(evaluator, dummy_models):
    # Create dummy inputs
    model = dummy_models[0]
    original_images = torch.randn(5, 3, 32, 32)
    true_labels = torch.randint(0, 10, (5,))
    adversarial_images = torch.randn(5, 3, 32, 32)

    # Mock the necessary methods
    evaluator._prepare_tensors = MagicMock(return_value=(
        original_images, adversarial_images, true_labels, None))
    evaluator._get_successful_on_source_mask = MagicMock(return_value=(
        torch.ones(5, dtype=torch.bool), adversarial_images, true_labels, None))
    evaluator._evaluate_transferability = MagicMock()

    evaluator.update(model, original_images, true_labels, adversarial_images)

    evaluator._prepare_tensors.assert_called_once()
    evaluator._get_successful_on_source_mask.assert_called_once()
    evaluator._evaluate_transferability.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_successful_on_source_mask(evaluator):
    model = DummyModel("test_model")
    original_images = torch.randn(1, 3, 32, 32)
    true_labels = torch.tensor([0])
    adversarial_images = torch.randn(1, 3, 32, 32)

    mask, filtered_adv_images, filtered_true_labels, filtered_target_labels = evaluator._get_successful_on_source_mask(
        model, original_images, true_labels, adversarial_images, False, None)

    assert mask.numel() == 1


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_evaluate_transferability(evaluator):
    adversarial_images = torch.randn(5, 3, 32, 32)
    true_labels = torch.randint(0, 10, (5,))
    successful_on_source_mask = torch.ones(5, dtype=torch.bool)

    evaluator._evaluate_model_transferability = MagicMock(
        return_value=torch.tensor(3))
    evaluator._move_tensors_to_cpu = MagicMock()

    evaluator._evaluate_transferability(
        adversarial_images, true_labels, successful_on_source_mask, False, None)

    evaluator._evaluate_model_transferability.assert_called()
    evaluator._move_tensors_to_cpu.assert_called()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_results(evaluator):
    evaluator.transferability_data = {
        'model_1': {'successful_transfer': 2, 'successful_on_source': 5},
        'model_2': {'successful_transfer': 3, 'successful_on_source': 5}
    }
    evaluator.total_successful_on_source = 10

    results = evaluator.get_results()
    assert results['model_1'] == 0.2
    assert results['model_2'] == 0.3
