import pytest

from advsecurenet.evaluation.evaluators.perturbation_effectiveness_evaluator import (
    PerturbationEffectivenessEvaluator,
)


@pytest.fixture
def evaluator():
    return PerturbationEffectivenessEvaluator()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_initialization(evaluator):
    assert evaluator.total_attack_success_rate == 0
    assert evaluator.total_perturbation_distance == 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_reset(evaluator):
    evaluator.total_attack_success_rate = 1.0
    evaluator.total_perturbation_distance = 1.0
    evaluator.reset()
    assert evaluator.total_attack_success_rate == 0
    assert evaluator.total_perturbation_distance == 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_update(evaluator):
    evaluator.update(attack_success_rate=0.8, perturbation_distance=0.5)
    assert evaluator.total_attack_success_rate == 0.8
    assert evaluator.total_perturbation_distance == 0.5

    evaluator.update(attack_success_rate=0.6, perturbation_distance=0.4)
    assert evaluator.total_attack_success_rate == 1.4
    assert evaluator.total_perturbation_distance == 0.9


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_results(evaluator):
    evaluator.update(attack_success_rate=0.8, perturbation_distance=0.5)
    evaluator.update(attack_success_rate=0.6, perturbation_distance=0.4)
    result = evaluator.get_results()
    assert result == pytest.approx(1.4 / 0.9, rel=1e-5)

    evaluator.reset()
    result = evaluator.get_results()
    assert result == 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_calculate_perturbation_effectiveness_score(evaluator):
    score = evaluator.calculate_perturbation_effectiveness_score(
        attack_success_rate=0.8, perturbation_distance=0.5
    )
    assert score == 0.8 / 0.5

    score = evaluator.calculate_perturbation_effectiveness_score(
        attack_success_rate=0.6, perturbation_distance=0.4
    )
    assert score == 0.6 / 0.4


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_results_with_zero_perturbation(evaluator):
    evaluator.update(attack_success_rate=0.8, perturbation_distance=0)
    result = evaluator.get_results()
    assert result == 0


if __name__ == "__main__":
    pytest.main()
