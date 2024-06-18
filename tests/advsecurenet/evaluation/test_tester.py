from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from advsecurenet.evaluation.tester import Tester
from advsecurenet.shared.types.configs.test_config import TestConfig


@pytest.fixture
def mock_model():
    model = MagicMock(spec=nn.Module)
    model._num_classes = 10
    return model


@pytest.fixture
def mock_test_loader():
    loader = MagicMock()
    data = torch.randn(5, 3, 32, 32)  # Example tensor for batch size 5
    target = torch.randint(0, 10, (5,))
    loader.__iter__.return_value = [(data, target)]
    loader.dataset = list(range(50))  # Pretend dataset with 50 samples
    return loader


@pytest.fixture
def mock_config(mock_model, mock_test_loader):
    return TestConfig(
        model=mock_model,
        test_loader=mock_test_loader,
        processor=torch.device("cpu"),
        criterion="cross_entropy",
        topk=5
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_tester_initialization(mock_config):
    tester = Tester(mock_config)
    assert tester._model == mock_config.model
    assert tester._test_loader == mock_config.test_loader
    assert tester._device == mock_config.processor
    assert isinstance(tester._loss_fn, nn.Module)
    assert tester._topk == mock_config.topk


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_tester_test_method(mock_config, mock_model):
    tester = Tester(mock_config)
    # Example output for batch size 5, 10 classes
    mock_model.return_value = torch.randn(5, 10)

    with patch("advsecurenet.evaluation.tester.tqdm", return_value=mock_config.test_loader):
        test_loss, accuracy_topk = tester.test()

    assert isinstance(test_loss, float)
    assert isinstance(accuracy_topk, float)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_validate_topk(mock_config, mock_model):
    mock_model.num_classes = 10
    mock_config.topk = 5
    tester = Tester(mock_config)
    tester._validate()

    mock_config.topk = 11
    with pytest.raises(ValueError, match="Top-k value must be less than or equal to the number of classes."):
        tester = Tester(mock_config)
        tester._validate()

    mock_config.topk = 0

    with pytest.raises(ValueError, match="Top-k value must be greater than 0."):
        tester = Tester(mock_config)
        tester._validate()
