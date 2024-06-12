import pytest
import torch

from advsecurenet.evaluation.evaluators.perturbation_distance_evaluator import \
    PerturbationDistanceEvaluator


@pytest.fixture
def evaluator():
    return PerturbationDistanceEvaluator()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_reset(evaluator):
    evaluator.total_l0_distance = 10
    evaluator.total_l2_distance = 20
    evaluator.total_l_inf_distance = 30
    evaluator.batch_size = 5

    evaluator.reset()

    assert evaluator.total_l0_distance == 0
    assert evaluator.total_l2_distance == 0
    assert evaluator.total_l_inf_distance == 0
    assert evaluator.batch_size == 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_update(evaluator):
    original_images = torch.zeros((5, 3, 32, 32))
    adversarial_images = torch.ones((5, 3, 32, 32))

    evaluator.update(original_images, adversarial_images)

    assert evaluator.total_l0_distance > 0
    assert evaluator.total_l2_distance > 0
    assert evaluator.total_l_inf_distance > 0
    assert evaluator.batch_size == 1


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_results(evaluator):
    original_images = torch.zeros((5, 3, 32, 32))
    adversarial_images = torch.ones((5, 3, 32, 32))

    evaluator.update(original_images, adversarial_images)
    results = evaluator.get_results()

    assert 'L0' in results
    assert 'L2' in results
    assert 'Linf' in results


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_perturbation_distance(evaluator):
    original_images = torch.zeros((5, 3, 32, 32))
    adversarial_images = torch.ones((5, 3, 32, 32))

    evaluator.update(original_images, adversarial_images)
    l0_distance = evaluator.get_perturbation_distance('L0')
    l2_distance = evaluator.get_perturbation_distance('L2')
    l_inf_distance = evaluator.get_perturbation_distance('Linf')

    assert l0_distance > 0
    assert l2_distance > 0
    assert l_inf_distance > 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_calculate_l0_distance(evaluator):
    original_images = torch.zeros((5, 3, 32, 32))
    adversarial_images = torch.ones((5, 3, 32, 32))

    l0_distance = evaluator.calculate_l0_distance(
        original_images, adversarial_images)

    assert l0_distance > 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_calculate_l2_distance(evaluator):
    original_images = torch.zeros((5, 3, 32, 32))
    adversarial_images = torch.ones((5, 3, 32, 32))

    l2_distance = evaluator.calculate_l2_distance(
        original_images, adversarial_images)

    assert l2_distance > 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_calculate_l_inf_distance(evaluator):
    original_images = torch.zeros((5, 3, 32, 32))
    adversarial_images = torch.ones((5, 3, 32, 32))

    l_inf_distance = evaluator.calculate_l_inf_distance(
        original_images, adversarial_images)

    assert l_inf_distance > 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_calculate_perturbation_distances(evaluator):
    original_images = torch.zeros((5, 3, 32, 32))
    adversarial_images = torch.ones((5, 3, 32, 32))

    l0_distance, l2_distance, l_inf_distance = evaluator.calculate_perturbation_distances(
        original_images, adversarial_images)

    assert l0_distance > 0
    assert l2_distance > 0
    assert l_inf_distance > 0
