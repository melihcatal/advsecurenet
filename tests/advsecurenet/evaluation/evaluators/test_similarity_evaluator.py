from unittest.mock import MagicMock, patch

import pytest
import torch

from advsecurenet.evaluation.evaluators.similarity_evaluator import SimilarityEvaluator


@pytest.fixture
def evaluator():
    return SimilarityEvaluator()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_initialization(evaluator):
    assert evaluator.ssim_score == 0
    assert evaluator.psnr_score == 0
    assert evaluator.total_images == 0
    assert evaluator.total_batches == 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_reset(evaluator):
    evaluator.ssim_score = 1.0
    evaluator.psnr_score = 1.0
    evaluator.total_images = 10
    evaluator.total_batches = 5
    evaluator.reset()
    assert evaluator.ssim_score == 0
    assert evaluator.psnr_score == 0
    assert evaluator.total_images == 0
    assert evaluator.total_batches == 0


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch(
    "advsecurenet.evaluation.evaluators.similarity_evaluator.SimilarityEvaluator.calculate_ssim",
    return_value=0.9,
)
def test_update_ssim(mock_calculate_ssim, evaluator):
    original_images = torch.zeros((2, 3, 32, 32))
    adversarial_images = torch.ones((2, 3, 32, 32))
    evaluator.update_ssim(original_images, adversarial_images)
    assert evaluator.ssim_score == 0.9
    assert evaluator.total_images == 2


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch(
    "advsecurenet.evaluation.evaluators.similarity_evaluator.SimilarityEvaluator.calculate_psnr",
    return_value=30.0,
)
def test_update_psnr(mock_calculate_psnr, evaluator):
    original_images = torch.zeros((2, 3, 32, 32))
    adversarial_images = torch.ones((2, 3, 32, 32))
    evaluator.update_psnr(original_images, adversarial_images)
    assert evaluator.psnr_score == 30.0
    assert evaluator.total_images == 2


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch(
    "advsecurenet.evaluation.evaluators.similarity_evaluator.SimilarityEvaluator.calculate_ssim",
    return_value=0.9,
)
@patch(
    "advsecurenet.evaluation.evaluators.similarity_evaluator.SimilarityEvaluator.calculate_psnr",
    return_value=30.0,
)
def test_update(mock_calculate_ssim, mock_calculate_psnr, evaluator):
    original_images = torch.zeros((2, 3, 32, 32))
    adversarial_images = torch.ones((2, 3, 32, 32))
    evaluator.update(original_images, adversarial_images)
    assert evaluator.ssim_score == 0.9
    assert evaluator.psnr_score == 30.0
    assert evaluator.total_batches == 1


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_ssim(evaluator):
    evaluator.ssim_score = 0.9
    assert evaluator.get_ssim() == 0.9


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_psnr(evaluator):
    evaluator.psnr_score = 30.0
    assert evaluator.get_psnr() == 30.0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_results(evaluator):
    evaluator.ssim_score = 0.9
    evaluator.psnr_score = 30.0
    evaluator.total_batches = 1
    results = evaluator.get_results()
    assert results["SSIM"] == 0.9
    assert results["PSNR"] == 30.0


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch(
    "advsecurenet.evaluation.evaluators.similarity_evaluator.ssim",
    return_value=torch.tensor([0.9]),
)
def test_calculate_ssim(mock_ssim, evaluator):
    original_images = torch.zeros((2, 3, 32, 32))
    adversarial_images = torch.ones((2, 3, 32, 32))
    ssim_score = evaluator.calculate_ssim(original_images, adversarial_images)
    assert round(ssim_score, 1) == 0.9


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch(
    "advsecurenet.evaluation.evaluators.similarity_evaluator.psnr",
    return_value=torch.tensor([30.0]),
)
def test_calculate_psnr(mock_psnr, evaluator):
    original_images = torch.zeros((2, 3, 32, 32))
    adversarial_images = torch.ones((2, 3, 32, 32))
    psnr_score = evaluator.calculate_psnr(original_images, adversarial_images)
    assert round(psnr_score, 1) == 30.0


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch(
    "advsecurenet.evaluation.evaluators.similarity_evaluator.SimilarityEvaluator.calculate_ssim",
    return_value=0.9,
)
@patch(
    "advsecurenet.evaluation.evaluators.similarity_evaluator.SimilarityEvaluator.calculate_psnr",
    return_value=30.0,
)
def test_calculate_similarity_scores(
    mock_calculate_ssim, mock_calculate_psnr, evaluator
):
    original_images = torch.zeros((2, 3, 32, 32))
    adversarial_images = torch.ones((2, 3, 32, 32))
    ssim_score, psnr_score = evaluator.calculate_similarity_scores(
        original_images, adversarial_images
    )
    assert round(ssim_score, 1) == 0.9
    assert round(psnr_score, 1) == 30.0
