from unittest.mock import MagicMock

import pytest
import torch

from advsecurenet.evaluation.evaluators.transferability_evaluator import \
    TransferabilityEvaluator


@pytest.fixture
def device(request):
    device_arg = request.config.getoption("--device")
    return torch.device(device_arg if device_arg else "cpu")


@pytest.fixture
def original_images(device):
    return torch.randn(10, 3, 32, 32, device=device)


@pytest.fixture
def adversarial_images(device):
    return torch.randn(10, 3, 32, 32, device=device)


@pytest.fixture
def true_labels(device):
    return torch.randint(0, 10, (10,), device=device)


@pytest.fixture
def target_labels(device):
    return torch.randint(0, 10, (10,), device=device)


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
def target_model(device):
    model = MagicMock()
    model.eval = MagicMock()
    param = torch.tensor([1.0], device=device)
    model.parameters = MagicMock(return_value=iter([param]))
    return model


@pytest.fixture
def successful_on_source_mask(device):
    return torch.randint(0, 2, (10,), dtype=torch.bool).to(device)


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
def test_evaluate_transferability_untargeted(evaluator):
    adversarial_images = torch.randn(5, 3, 32, 32)
    true_labels = torch.randint(0, 10, (5,))
    successful_on_source_mask = torch.ones(5, dtype=torch.bool)

    evaluator._evaluate_model_transferability = MagicMock(
        return_value=torch.tensor(3))
    evaluator._move_tensors_to_cpu = MagicMock()
    evaluator._move_tensors_to_cpu.return_value = (
        adversarial_images, true_labels, successful_on_source_mask)

    evaluator._evaluate_transferability(
        adversarial_images, true_labels, successful_on_source_mask, False, None)

    evaluator._evaluate_model_transferability.assert_called()
    evaluator._move_tensors_to_cpu.assert_called()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_evaluate_transferabilit_targeted(evaluator):
    adversarial_images = torch.randn(5, 3, 32, 32)
    true_labels = torch.randint(0, 10, (5,))
    target_labels = torch.randint(0, 10, (5,))
    successful_on_source_mask = torch.ones(5, dtype=torch.bool)

    evaluator._evaluate_model_transferability = MagicMock(
        return_value=torch.tensor(3))
    evaluator._move_tensors_to_cpu = MagicMock()
    evaluator._move_tensors_to_cpu.return_value = (
        adversarial_images, true_labels, successful_on_source_mask, target_labels)

    evaluator._evaluate_transferability(
        adversarial_images, true_labels, successful_on_source_mask, True, target_labels)

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


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_prepare_tensors_non_targeted(evaluator, device, original_images, adversarial_images, true_labels):
    result = evaluator._prepare_tensors(
        device, original_images, adversarial_images, true_labels, is_targeted=False, target_labels=None
    )

    assert result[0].device.type == device.type
    assert result[1].device.type == device.type
    assert result[2].device.type == device.type
    assert result[3] is None


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_prepare_tensors_targeted(evaluator, device, original_images, adversarial_images, true_labels, target_labels):
    result = evaluator._prepare_tensors(
        device, original_images, adversarial_images, true_labels, is_targeted=True, target_labels=target_labels
    )

    assert result[0].device.type == device.type
    assert result[1].device.type == device.type
    assert result[2].device.type == device.type
    assert result[3].device.type == device.type


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_prepare_tensors_targeted_no_target_labels(evaluator, device, original_images, adversarial_images, true_labels):
    with pytest.raises(ValueError, match="Target labels must be provided for targeted attacks."):
        evaluator._prepare_tensors(
            device, original_images, adversarial_images, true_labels, is_targeted=True, target_labels=None
        )


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_evaluate_model_transferability_non_targeted(evaluator, target_model, adversarial_images, true_labels, successful_on_source_mask):
    target_model.return_value = torch.randn(
        10, 10).to(adversarial_images.device.type)

    result = evaluator._evaluate_model_transferability(
        target_model, adversarial_images, true_labels, successful_on_source_mask, is_targeted=False, target_labels=None
    )

    target_model.assert_called_once_with(adversarial_images)
    assert result.device.type == adversarial_images.device.type
    assert result.dtype == torch.int64


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_evaluate_model_transferability_targeted(evaluator, target_model, adversarial_images, true_labels, successful_on_source_mask, target_labels):
    target_model.return_value = torch.randn(
        10, 10).to(adversarial_images.device.type)

    result = evaluator._evaluate_model_transferability(
        target_model, adversarial_images, true_labels, successful_on_source_mask, is_targeted=True, target_labels=target_labels
    )

    target_model.assert_called_once_with(adversarial_images)
    assert result.device.type == adversarial_images.device.type
    assert result.dtype == torch.int64


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_evaluate_model_transferability_no_successful(evaluator, target_model, adversarial_images, true_labels):
    successful_on_source_mask = torch.zeros(
        10, dtype=torch.bool).to(adversarial_images.device.type)
    target_model.return_value = torch.randn(
        10, 10).to(adversarial_images.device.type)

    result = evaluator._evaluate_model_transferability(
        target_model, adversarial_images, true_labels, successful_on_source_mask, is_targeted=False, target_labels=None
    )

    target_model.assert_called_once_with(adversarial_images)
    assert result == 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_move_tensors_to_cpu_targeted(evaluator, adversarial_images, true_labels, successful_on_source_mask, target_labels):
    adversarial_images, true_labels, successful_on_source_mask, target_labels = evaluator._move_tensors_to_cpu(
        adversarial_images, true_labels, successful_on_source_mask, target_labels)

    assert adversarial_images.device == torch.device("cpu")
    assert true_labels.device == torch.device("cpu")
    assert successful_on_source_mask.device == torch.device("cpu")
    assert target_labels.device == torch.device("cpu")


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_move_tensors_to_cpu_targeted_no_target_labels(evaluator, adversarial_images, true_labels, successful_on_source_mask):
    target_labels = None
    adversarial_images, true_labels, successful_on_source_mask = evaluator._move_tensors_to_cpu(
        adversarial_images, true_labels, successful_on_source_mask, target_labels)

    assert adversarial_images.device == torch.device("cpu")
    assert true_labels.device == torch.device("cpu")
    assert successful_on_source_mask.device == torch.device("cpu")
    assert target_labels is None
