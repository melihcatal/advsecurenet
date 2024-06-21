import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from advsecurenet.utils.model_utils import (download_weights, load_model,
                                            save_model)


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_save_model():
    model = SimpleModel()
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = "test_model"
        save_model(model, filename, filepath=temp_dir)
        assert os.path.exists(os.path.join(temp_dir, f"{filename}.pth"))


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_load_model():
    model = SimpleModel()
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = "test_model"
        save_model(model, filename, filepath=temp_dir)
        loaded_model = SimpleModel()
        load_model(loaded_model, filename, filepath=temp_dir)
        for param1, param2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.equal(param1, param2)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_save_model_distributed():
    model = nn.DataParallel(SimpleModel())
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = "test_model_distributed"
        save_model(model, filename, filepath=temp_dir, distributed=True)
        assert os.path.exists(os.path.join(temp_dir, f"{filename}.pth"))


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.utils.model_utils.requests.get')
def test_download_weights(mock_get):
    model_name = "resnet50"
    dataset_name = "cifar10"
    filename = f"{model_name}_{dataset_name}_weights.pth"

    # Mock the response of the requests.get call
    mock_response = MagicMock()
    mock_response.iter_content = MagicMock(return_value=[b'1234'])
    mock_response.headers = {'content-length': '4'}
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    with tempfile.TemporaryDirectory() as temp_dir:
        download_weights(model_name=model_name,
                         dataset_name=dataset_name, save_path=temp_dir)
        assert os.path.exists(os.path.join(temp_dir, filename))


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_download_weights_already_exists():
    model_name = "resnet50"
    dataset_name = "cifar10"
    filename = f"{model_name}_{dataset_name}_weights.pth"

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a file to simulate already downloaded weights
        with open(os.path.join(temp_dir, filename), 'w') as f:
            f.write("dummy content")

        with patch('advsecurenet.utils.model_utils.requests.get') as mock_get:
            download_weights(model_name=model_name,
                             dataset_name=dataset_name, save_path=temp_dir)
            mock_get.assert_not_called()
