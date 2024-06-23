from collections import deque
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import torch

from advsecurenet.attacks.decision_based.boundary import DecisionBoundary
from advsecurenet.shared.types.configs import attack_configs
from advsecurenet.utils.device_manager import DeviceManager

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
    attack_instance._perturb_orthogonal = MagicMock(
        return_value=(dummy_images, dummy_config.initial_delta))
    attack_instance._perturb_forward = MagicMock(
        return_value=(dummy_images, dummy_config.initial_epsilon))
    attack_instance._update_best_images = MagicMock(
        return_value=(dummy_images, torch.tensor([0.5])))

    # Call the attack method
    adversarial_images = attack_instance.attack(
        dummy_model, dummy_images, dummy_labels)

    # Assertions
    assert adversarial_images.shape == dummy_images.shape
    assert isinstance(adversarial_images, torch.Tensor)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_orthogonal_perturb(attack_instance):
    delta = 0.1
    perturbed = attack_instance._orthogonal_perturb(
        delta, dummy_images, dummy_images)
    assert perturbed.shape == dummy_images.shape
    assert isinstance(perturbed, torch.Tensor)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_forward_perturb(attack_instance):
    epsilon = 0.1
    perturbed = attack_instance._forward_perturb(
        epsilon, dummy_images, dummy_images)
    assert perturbed.shape == dummy_images.shape
    assert isinstance(perturbed, torch.Tensor)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_update_adv_images_all_success_true(attack_instance):
    success = torch.tensor([True, True, True])
    adv_images = torch.tensor([[1, 1], [2, 2], [3, 3]])
    trial_images = torch.tensor([[10, 10], [20, 20], [30, 30]])
    expected = trial_images.clone()

    result = attack_instance._update_adv_images(
        success, adv_images, trial_images)
    assert torch.equal(
        result, expected), "All elements should be updated with trial_images."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_update_adv_images_all_success_false(attack_instance):
    success = torch.tensor([False, False, False])
    adv_images = torch.tensor([[1, 1], [2, 2], [3, 3]])
    trial_images = torch.tensor([[10, 10], [20, 20], [30, 30]])
    expected = adv_images.clone()

    result = attack_instance._update_adv_images(
        success, adv_images, trial_images)
    assert torch.equal(result, expected), "No elements should be updated."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_update_adv_images_mixed_success(attack_instance):
    success = torch.tensor([True, False, True])
    adv_images = torch.tensor([[1, 1], [2, 2], [3, 3]])
    trial_images = torch.tensor([[10, 10], [20, 20], [30, 30]])
    expected = torch.tensor([[10, 10], [2, 2], [30, 30]])

    result = attack_instance._update_adv_images(
        success, adv_images, trial_images)
    assert torch.equal(
        result, expected), "Only elements where success is True should be updated."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_update_adv_images_empty_tensors(attack_instance):
    success = torch.tensor([])
    adv_images = torch.tensor([])
    trial_images = torch.tensor([])
    expected = torch.tensor([])

    result = attack_instance._update_adv_images(
        success, adv_images, trial_images)
    assert torch.equal(
        result, expected), "Empty tensors should return empty tensors."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_update_adv_images_single_element_tensors(attack_instance):
    success = torch.tensor([True])
    adv_images = torch.tensor([[1, 1]])
    trial_images = torch.tensor([[10, 10]])
    expected = torch.tensor([[10, 10]])

    result = attack_instance._update_adv_images(
        success, adv_images, trial_images)
    assert torch.equal(
        result, expected), "Single element tensors should be updated correctly."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_update_adv_images_single_element_false(attack_instance):
    success = torch.tensor([False])
    adv_images = torch.tensor([[1, 1]])
    trial_images = torch.tensor([[10, 10]])
    expected = torch.tensor([[1, 1]])

    result = attack_instance._update_adv_images(
        success, adv_images, trial_images)
    assert torch.equal(
        result, expected), "Single element tensors should remain unchanged if success is False."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_evaluate_success_targeted(attack_instance):
    predictions = torch.tensor([0, 1, 2])
    y = torch.tensor([0, 1, 2])
    expected = torch.tensor([True, True, True])

    is_targeted = True
    attack_instance.targeted = is_targeted

    returned_suceess = attack_instance._evaluate_success(predictions, y)

    assert torch.equal(
        returned_suceess, expected), "All predictions should be successful."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_evaluate_success_untargeted(attack_instance):
    predictions = torch.tensor([0, 1, 2])
    y = torch.tensor([0, 1, 2])
    expected = torch.tensor([False, False, False])

    is_targeted = False
    attack_instance.targeted = is_targeted

    returned_suceess = attack_instance._evaluate_success(predictions, y)

    assert torch.equal(
        returned_suceess, expected), "All predictions should be successful."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adjust_delta_success_rate_less_than_0_5(attack_instance):
    success = torch.tensor([False, False, True, False])
    delta = 1.0
    expected_delta = delta * attack_instance.step_adapt

    result = attack_instance._adjust_delta(success, delta)
    assert result == expected_delta, "Delta should be multiplied by step_adapt."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adjust_delta_success_rate_equal_0_5(attack_instance):
    success = torch.tensor([False, True, True, False])
    delta = 1.0
    expected_delta = delta / attack_instance.step_adapt

    result = attack_instance._adjust_delta(success, delta)
    assert result == expected_delta, "Delta should be divided by step_adapt."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adjust_delta_success_rate_greater_than_0_5(attack_instance):
    success = torch.tensor([True, True, True, False])
    delta = 1.0
    expected_delta = delta / attack_instance.step_adapt

    result = attack_instance._adjust_delta(success, delta)
    assert result == expected_delta, "Delta should be divided by step_adapt."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adjust_delta_empty_tensor(attack_instance):
    success = torch.tensor([])
    delta = 1.0

    result = attack_instance._adjust_delta(success, delta)
    assert result == delta, "Empty success tensor should result in delta unchanged."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adjust_delta_single_element_tensor(attack_instance):
    success = torch.tensor([True])
    delta = 1.0
    expected_delta = delta / attack_instance.step_adapt

    result = attack_instance._adjust_delta(success, delta)
    assert result == expected_delta, "Single True element should result in delta divided by step_adapt."

    success = torch.tensor([False])
    expected_delta = delta * attack_instance.step_adapt

    result = attack_instance._adjust_delta(success, delta)
    assert result == expected_delta, "Single False element should result in delta multiplied by step_adapt."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adjust_epsilon_success_rate_less_than_0_5(attack_instance):
    success = torch.tensor([False, False, True, False])
    epsilon = 1.0
    expected_epsilon = epsilon * attack_instance.step_adapt

    result = attack_instance._adjust_epsilon(success, epsilon)
    assert result == expected_epsilon, "Epsilon should be multiplied by step_adapt when success rate is less than 0.5."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adjust_epsilon_success_rate_equal_0_5(attack_instance):
    success = torch.tensor([False, True, True, False])
    epsilon = 1.0
    expected_epsilon = epsilon / attack_instance.step_adapt

    result = attack_instance._adjust_epsilon(success, epsilon)
    assert result == expected_epsilon, "Epsilon should be divided by step_adapt when success rate is equal to 0.5."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adjust_epsilon_success_rate_greater_than_0_5(attack_instance):
    success = torch.tensor([True, True, True, False])
    epsilon = 1.0
    expected_epsilon = epsilon / attack_instance.step_adapt

    result = attack_instance._adjust_epsilon(success, epsilon)
    assert result == expected_epsilon, "Epsilon should be divided by step_adapt when success rate is greater than 0.5."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adjust_epsilon_empty_tensor(attack_instance):
    success = torch.tensor([])
    epsilon = 1.0
    expected_epsilon = epsilon

    result = attack_instance._adjust_epsilon(success, epsilon)
    assert result == expected_epsilon, "Empty success tensor should return the same epsilon value."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adjust_epsilon_single_element_tensor(attack_instance):
    success = torch.tensor([True])
    epsilon = 1.0
    expected_epsilon = epsilon / attack_instance.step_adapt

    result = attack_instance._adjust_epsilon(success, epsilon)
    assert result == expected_epsilon, "Single True element should result in epsilon divided by step_adapt."

    success = torch.tensor([False])
    expected_epsilon = epsilon * attack_instance.step_adapt

    result = attack_instance._adjust_epsilon(success, epsilon)
    assert result == expected_epsilon, "Single False element should result in epsilon multiplied by step_adapt."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_update_best_images(attack_instance):
    adv_images = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]]])
    x = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[1.0, 1.0], [1.0, 1.0]]]])
    best_adv_images = torch.tensor(
        [[[[0.5, 0.5], [0.5, 0.5]]], [[[2.0, 2.0], [2.0, 2.0]]]])
    best_distances = torch.tensor([10.0, 10.0])

    expected_best_adv_images = torch.tensor([[[[1., 2.],
                                               [3., 4.]]],
                                             [[[2., 2.],
                                               [2., 2.]]]])
    expected_best_distances = torch.tensor([5.4772, 10.0000])

    result_best_adv_images, result_best_distances = attack_instance._update_best_images(
        adv_images, x, best_adv_images, best_distances)

    assert torch.equal(result_best_adv_images,
                       expected_best_adv_images), "Best adversarial images should be updated."

    assert torch.allclose(result_best_distances, expected_best_distances,
                          atol=1e-4), "Best distances should be updated."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_no_update_best_images(attack_instance):
    adv_images = torch.tensor(
        [[[[1.0, 1.0], [1.0, 1.0]]], [[[2.0, 2.0], [2.0, 2.0]]]])
    x = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]])
    best_adv_images = torch.tensor(
        [[[[0.5, 0.5], [0.5, 0.5]]], [[[1.0, 1.0], [1.0, 1.0]]]])
    best_distances = torch.tensor([2.0, 3.0])

    expected_best_adv_images = best_adv_images.clone()
    expected_best_distances = best_distances.clone()

    result_best_adv_images, result_best_distances = attack_instance._update_best_images(
        adv_images, x, best_adv_images, best_distances)

    assert torch.equal(result_best_adv_images,
                       expected_best_adv_images), "Best adversarial images should not be updated."
    assert torch.equal(result_best_distances,
                       expected_best_distances), "Best distances should not be updated."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_partial_update_best_images(attack_instance):
    adv_images = torch.tensor(
        [[[[1.0, 1.0], [1.0, 1.0]]], [[[0.5, 0.5], [0.5, 0.5]]]])
    x = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]])
    best_adv_images = torch.tensor(
        [[[[0.5, 0.5], [0.5, 0.5]]], [[[2.0, 2.0], [2.0, 2.0]]]])
    best_distances = torch.tensor([2.0, 1.0])

    expected_best_adv_images = torch.tensor([[[[0.5000, 0.5000],
                                               [0.5000, 0.5000]]],

                                             [[[2.0000, 2.0000],
                                               [2.0000, 2.0000]]]])

    expected_best_distances = torch.tensor([2., 1.])

    result_best_adv_images, result_best_distances = attack_instance._update_best_images(
        adv_images, x, best_adv_images, best_distances)

    assert torch.allclose(result_best_adv_images, expected_best_adv_images,
                          atol=1e-4), "Best adversarial images should be partially updated."
    assert torch.allclose(result_best_distances, expected_best_distances,
                          atol=1e-4), "Best distances should be partially updated."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_perturb_orthogonal(attack_instance):
    model = MagicMock()
    model.return_value = torch.tensor([[0.1, 0.9], [0.3, 0.7]])
    attack_instance._orthogonal_perturb = MagicMock(
        return_value=torch.tensor([[[[0.1]]], [[[0.1]]]]))
    attack_instance._evaluate_success = MagicMock(
        return_value=torch.tensor([True, False]))
    attack_instance._adjust_delta = MagicMock(return_value=0.5)
    attack_instance._update_adv_images = MagicMock(
        return_value=torch.tensor([[[[0.5]]], [[[0.5]]]]))

    x = torch.tensor([[[[0.0]]], [[[0.0]]]])
    y = torch.tensor([1, 0])
    adv_images = torch.tensor([[[[0.4]]], [[[0.4]]]])
    delta = 1.0

    result_adv_images, result_delta = attack_instance._perturb_orthogonal(
        model, x, y, adv_images, delta)

    assert attack_instance._orthogonal_perturb.call_count == attack_instance.max_delta_trials, "The _orthogonal_perturb method should be called max_delta_trials times."
    assert model.call_count == attack_instance.max_delta_trials, "The model should be called max_delta_trials times."
    assert attack_instance._evaluate_success.call_count == attack_instance.max_delta_trials, "The _evaluate_success method should be called max_delta_trials times."
    assert attack_instance._adjust_delta.call_count == attack_instance.max_delta_trials, "The _adjust_delta method should be called max_delta_trials times."
    assert attack_instance._update_adv_images.call_count == attack_instance.max_delta_trials, "The _update_adv_images method should be called max_delta_trials times."

    expected_adv_images = torch.tensor([[[[0.5]]], [[[0.5]]]])
    expected_delta = 0.5

    assert torch.equal(
        result_adv_images, expected_adv_images), "The resulting adversarial images should match the expected value."
    assert result_delta == expected_delta, "The resulting delta should match the expected value."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_perturb_forward(attack_instance):
    model = MagicMock()
    model.return_value = torch.tensor([[0.1, 0.9], [0.3, 0.7]])
    attack_instance._forward_perturb = MagicMock(
        return_value=torch.tensor([[[[0.1]]], [[[0.1]]]]))
    attack_instance._evaluate_success = MagicMock(
        return_value=torch.tensor([True, False]))
    attack_instance._adjust_epsilon = MagicMock(return_value=0.5)
    attack_instance._update_adv_images = MagicMock(
        return_value=torch.tensor([[[[0.5]]], [[[0.5]]]]))

    x = torch.tensor([[[[0.0]]], [[[0.0]]]])
    y = torch.tensor([1, 0])
    adv_images = torch.tensor([[[[0.4]]], [[[0.4]]]])
    epsilon = 1.0

    result_adv_images, result_epsilon = attack_instance._perturb_forward(
        model, x, y, adv_images, epsilon)

    assert attack_instance._forward_perturb.call_count == attack_instance.max_epsilon_trials, "The _forward_perturb method should be called max_epsilon_trials times."
    assert model.call_count == attack_instance.max_epsilon_trials, "The model should be called max_epsilon_trials times."
    assert attack_instance._evaluate_success.call_count == attack_instance.max_epsilon_trials, "The _evaluate_success method should be called max_epsilon_trials times."
    assert attack_instance._adjust_epsilon.call_count == attack_instance.max_epsilon_trials, "The _adjust_epsilon method should be called max_epsilon_trials times."
    assert attack_instance._update_adv_images.call_count == attack_instance.max_epsilon_trials, "The _update_adv_images method should be called max_epsilon_trials times."

    expected_adv_images = torch.tensor([[[[0.5]]], [[[0.5]]]])
    expected_epsilon = 0.5

    assert torch.equal(
        result_adv_images, expected_adv_images), "The resulting adversarial images should match the expected value."
    assert result_epsilon == expected_epsilon, "The resulting epsilon should match the expected value."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_check_early_stopping_no_early_stopping_before_patience(attack_instance):
    best_distances = torch.tensor([1.0, 1.5, 2.0])
    recent_improvements = deque([1.6, 1.4])
    iteration = attack_instance.early_stopping_patience - 1

    result = attack_instance._check_early_stopping(
        best_distances, recent_improvements, iteration)
    assert not result, "Early stopping should not trigger before reaching patience."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_check_early_stopping_no_early_stopping_with_improvement(attack_instance):
    best_distances = torch.tensor([1.0, 1.5, 2.0])
    recent_improvements = deque([1.6, 1.4])
    iteration = 3
    improvement = recent_improvements[0] - best_distances.mean().item()

    attack_instance.early_stopping_threshold = improvement - 0.01

    result = attack_instance._check_early_stopping(
        best_distances, recent_improvements, iteration)
    assert not result, "Early stopping should not trigger when there is sufficient improvement."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_check_early_stopping_early_stopping_triggered(attack_instance):

    best_distances = torch.tensor([1.0, 1.5, 2.0])
    recent_improvements = deque([1.6, 1.59])
    improvement = recent_improvements[0] - best_distances.mean().item()

    attack_instance.early_stopping_threshold = improvement + 0.01
    iteration = attack_instance.early_stopping_patience + 1

    result = attack_instance._check_early_stopping(
        best_distances, recent_improvements, iteration)
    assert result, "Early stopping should trigger when improvement is below threshold."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_check_early_stopping_early_stopping_with_verbose(attack_instance, capsys):
    attack_instance.verbose = True
    best_distances = torch.tensor([1.0, 1.5, 2.0])
    recent_improvements = deque([1.6, 1.59])
    iteration = attack_instance.early_stopping_patience + 1
    improvement = recent_improvements[0] - best_distances.mean().item()

    attack_instance.early_stopping_threshold = improvement + 0.01
    result = attack_instance._check_early_stopping(
        best_distances, recent_improvements, iteration)
    captured = capsys.readouterr()
    assert result, "Early stopping should trigger when improvement is below threshold."
    assert "Early stopping" in captured.out, "Verbose message should be printed when early stopping is triggered."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_initialize_untargeted(attack_instance):
    model = MagicMock()
    model.return_value = torch.tensor([[0.1, 0.9], [0.7, 0.3]])
    x = torch.tensor([[[[0.1]]], [[[0.2]]]])
    y = torch.tensor([1, 0])

    perturbed_images = attack_instance._initialize(model, x, y)

    assert perturbed_images.shape == x.shape, "Perturbed images should have the same shape as input images."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_initialize_targeted(attack_instance):
    attack_instance.targeted = True
    model = MagicMock()
    model.return_value = torch.tensor([[0.1, 0.9], [0.7, 0.3]])
    x = torch.tensor([[[[0.1]]], [[[0.2]]]])
    y = torch.tensor([1, 0])

    perturbed_images = attack_instance._initialize(model, x, y)

    assert perturbed_images.shape == x.shape, "Perturbed images should have the same shape as input images."


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_initialize_device_manager(attack_instance):
    attack_instance.device_manager = DeviceManager(
        device="cpu", distributed_mode=False)
    model = MagicMock()
    model.return_value = torch.tensor([[0.1, 0.9], [0.7, 0.3]])
    x = torch.tensor([[[[0.1]]], [[[0.2]]]])
    y = torch.tensor([1, 0])

    assert x.device == torch.device(
        'cpu'), "Input tensor should be on the correct device."
    assert y.device == torch.device(
        'cpu'), "Input tensor should be on the correct device."
