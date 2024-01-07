import pytest
import torch

from advsecurenet.evaluation.evaluators.similarity_evaluator import \
    SimilarityEvaluator


# Helper function to create mock data
def create_mock_images(batch_size, channels, height, width):
    return torch.rand(batch_size, channels, height, width)


@pytest.fixture
def evaluator():
    return SimilarityEvaluator()


def test_initialization(evaluator):
    assert evaluator.ssim_score == 0
    assert evaluator.psnr_score == 0
    assert evaluator.total_images == 0
    assert evaluator.total_batches == 0


def test_reset(evaluator):
    evaluator.reset()
    assert evaluator.ssim_score == 0
    assert evaluator.psnr_score == 0
    assert evaluator.total_images == 0
    assert evaluator.total_batches == 0


def test_update_ssim_psnr(evaluator):
    original_images = create_mock_images(10, 3, 224, 224)
    adversarial_images = create_mock_images(10, 3, 224, 224)

    evaluator.update_ssim(original_images, adversarial_images)
    assert evaluator.ssim_score > 0
    assert evaluator.total_images == 10

    evaluator.update_psnr(original_images, adversarial_images)
    assert evaluator.psnr_score > 0
    assert evaluator.total_images == 20


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
