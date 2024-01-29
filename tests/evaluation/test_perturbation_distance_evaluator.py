import pytest
import torch

from advsecurenet.evaluation.evaluators.perturbation_distance_evaluator import \
    PerturbationDistanceEvaluator


# Helper function to create mock data
def create_mock_images(batch_size, channels, height, width):
    return torch.rand(batch_size, channels, height, width)


@pytest.fixture
def evaluator():
    return PerturbationDistanceEvaluator()


def test_initialization(evaluator):
    assert evaluator.total_l0_distance == 0
    assert evaluator.total_l2_distance == 0
    assert evaluator.total_l_inf_distance == 0
    assert evaluator.batch_size == 0


def test_reset(evaluator):
    evaluator.reset()
    assert evaluator.total_l0_distance == 0
    assert evaluator.total_l2_distance == 0
    assert evaluator.total_l_inf_distance == 0
    assert evaluator.batch_size == 0


def test_update(evaluator):
    original_images = create_mock_images(10, 3, 224, 224)
    adversarial_images = create_mock_images(10, 3, 224, 224)

    evaluator.update(original_images, adversarial_images)
    assert evaluator.batch_size == 1
    assert evaluator.total_l0_distance > 0
    assert evaluator.total_l2_distance > 0
    assert evaluator.total_l_inf_distance > 0


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
