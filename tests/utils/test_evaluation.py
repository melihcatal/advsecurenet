# test_adversarial_attack_evaluator.py

import pytest
import torch
import numpy as np
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.utils.evaluation import AdversarialAttackEvaluator


@pytest.fixture
def dummy_model():
    model = ModelFactory.create_model(
        "CustomMnistModel", num_classes=10, num_input_channels=1)
    return model


@pytest.fixture
def evaluator(dummy_model):
    return AdversarialAttackEvaluator(dummy_model)


def test_initialization(dummy_model):
    evaluator = AdversarialAttackEvaluator(dummy_model)
    assert evaluator.model == dummy_model


def test_full_evaluation(evaluator):
    # Set up dummy data
    original_images = torch.rand((10, 1, 28, 28))
    true_labels = torch.randint(0, 10, (10,))
    adversarial_images = torch.rand((10, 1, 28, 28))
    target_labels = torch.randint(0, 10, (10,))

    # Test non-targeted attack
    results = evaluator.full_evaluation(
        original_images, true_labels, adversarial_images)
    assert isinstance(results, dict)

    # Test targeted attack
    results = evaluator.full_evaluation(
        original_images, true_labels, adversarial_images, is_targeted=True, target_labels=target_labels)
    assert isinstance(results, dict)


def test_calculate_l0_distance(evaluator):
    # Create test data
    original_images = torch.rand((10, 1, 28, 28))
    adversarial_images = torch.clone(original_images)
    adversarial_images[0, 0, 0, 0] = 0  # Change one pixel

    l0_distance = evaluator.calculate_l0_distance(
        original_images, adversarial_images)
    expected_l0_distance = 1 / 10  # Only one pixel in one image out of 10 is changed
    assert l0_distance == pytest.approx(expected_l0_distance, rel=1e-3)


def test_calculate_l2_distance(evaluator):
    # Create test data
    original_images = torch.rand((10, 1, 28, 28))
    adversarial_images = torch.clone(original_images)
    adversarial_images[0, 0, 0, 0] += 0.5  # Increase one pixel value

    l2_distance = evaluator.calculate_l2_distance(
        original_images, adversarial_images)
    expected_l2_distance = torch.norm((original_images - adversarial_images).view(
        original_images.shape[0], -1), p=2, dim=1).mean().item()
    assert l2_distance == pytest.approx(expected_l2_distance, rel=1e-3)


def test_calculate_l_inf_distance(evaluator):
    # Create test data
    original_images = torch.rand((10, 1, 28, 28))
    adversarial_images = torch.clone(original_images)
    adversarial_images[0, 0, 0, 0] += 0.5  # Increase one pixel value

    l_inf_distance = evaluator.calculate_l_inf_distance(
        original_images, adversarial_images)
    # maximum absolute difference between two images
    expected_l_inf_distance = (original_images - adversarial_images).view(
        original_images.shape[0], -1).abs().max(dim=1)[0].mean().item()
    assert l_inf_distance == pytest.approx(expected_l_inf_distance, rel=1e-3)


def test_identical_images_distance(evaluator):
    # Create test data
    original_images = torch.rand((10, 1, 28, 28))
    adversarial_images = torch.clone(original_images)

    l0_distance = evaluator.calculate_l0_distance(
        original_images, adversarial_images)
    expected_l0_distance = 0
    assert l0_distance == pytest.approx(expected_l0_distance, rel=1e-3)

    l2_distance = evaluator.calculate_l2_distance(
        original_images, adversarial_images)
    expected_l2_distance = 0
    assert l2_distance == pytest.approx(expected_l2_distance, rel=1e-3)

    l_inf_distance = evaluator.calculate_l_inf_distance(
        original_images, adversarial_images)
    expected_l_inf_distance = 0
    assert l_inf_distance == pytest.approx(expected_l_inf_distance, rel=1e-3)


def test_identical_images_ssim(evaluator):
    # Create identical original and adversarial images
    original = torch.rand((10, 3, 224, 224))
    adversarial = original.clone()

    ssim = evaluator.calculate_ssim(original, adversarial)

    assert ssim == pytest.approx(1.0, rel=1e-3)


def test_identical_images_psnr(evaluator):
    # Create identical original and adversarial images
    original = torch.rand((10, 3, 224, 224))
    adversarial = original.clone()

    psnr = evaluator.calculate_psnr(original, adversarial)

    assert psnr == pytest.approx(float("inf"), rel=1e-3)
