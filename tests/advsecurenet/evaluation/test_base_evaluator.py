import csv
import os
from datetime import datetime
from unittest.mock import MagicMock, mock_open, patch

import pytest
import torch

from advsecurenet.evaluation.base_evaluator import BaseEvaluator
from advsecurenet.models.base_model import BaseModel


class MockModel(BaseModel):
    def forward(self, x):
        return x

    def load_model(self):
        pass

    def models(self):
        return [self]


class ConcreteEvaluator(BaseEvaluator):
    def __init__(self):
        super(ConcreteEvaluator, self).__init__()
        self.results = []

    def reset(self):
        self.results = []

    def update(self, model, original_images, true_labels, adversarial_images, is_targeted=False, target_labels=None):
        self.results.append((original_images, adversarial_images))

    def get_results(self):
        return {"accuracy": 0.95, "robustness": 0.85}


@pytest.fixture
def evaluator():
    return ConcreteEvaluator()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_reset(evaluator):
    evaluator.reset()
    assert evaluator.results == []


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_update(evaluator):
    model = MockModel()
    original_images = torch.randn(10, 3, 32, 32)
    true_labels = torch.randint(0, 10, (10,))
    adversarial_images = torch.randn(10, 3, 32, 32)

    evaluator.update(model, original_images, true_labels, adversarial_images)
    assert len(evaluator.results) == 1


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_results(evaluator):
    results = evaluator.get_results()
    assert "accuracy" in results
    assert "robustness" in results


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("builtins.open", new_callable=mock_open)
@patch("os.makedirs")
def test_save_results_to_csv(mock_makedirs, mock_open, evaluator):
    evaluation_results = {"accuracy": 0.95, "robustness": 0.85}
    experiment_info = {"model": "MockModel", "attack": "MockAttack"}
    path = "test_dir"
    file_name = "test_file.csv"

    evaluator.save_results_to_csv(
        evaluation_results, experiment_info, path, file_name)

    mock_makedirs.assert_called_once_with(path, exist_ok=True)
    mock_open.assert_called_once_with(os.path.join(
        path, file_name), mode='a', newline='', encoding='utf-8')

    handle = mock_open()
    writer = csv.writer(handle)

    writer.writerow.assert_any_call(
        ["Experiment conducted on " + datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    for key, value in experiment_info.items():
        writer.writerow.assert_any_call([key, value])
    writer.writerow.assert_any_call(["-" * 10, "-" * 10])
    writer.writerow.assert_any_call(list(evaluation_results.keys()))
    writer.writerow.assert_any_call([str(value)
                                    for value in evaluation_results.values()])


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("builtins.open", new_callable=mock_open)
@patch("os.makedirs")
def test_save_results_to_csv(mock_makedirs, mock_open, evaluator):
    evaluation_results = {"accuracy": 0.95, "robustness": 0.85}
    evaluator.save_results_to_csv(evaluation_results)

    file_name = datetime.now().strftime('%Y%m%d_%H%M%S') + "_experiment.csv"
    mock_open.assert_called_once_with(
        file_name, mode='a', newline='', encoding='utf-8')


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_context_management(evaluator):
    with evaluator as e:
        assert isinstance(e, ConcreteEvaluator)
        assert e.results == []

    assert evaluator.results == []
