import pytest
from torch import nn
import torch
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader, TensorDataset
from advsecurenet.dataloader.data_loader_factory import DataLoaderFactory
from advsecurenet.datasets.dataset_factory import DatasetFactory
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types.configs.train_config import TrainConfig
from advsecurenet.shared.types.dataset import DatasetType
from advsecurenet.utils.model_utils import save_model, load_model, download_weights


# Fixture for model
@pytest.fixture
def model():
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc = nn.Linear(10, 10)  # Adding a linear layer

        def forward(self, x):
            return self.fc(x)  # Use the linear layer in forward
    return DummyModel()


# Fixture for data loader
@pytest.fixture
def dataloader():
    data = torch.rand(100, 10)
    labels = torch.randint(0, 2, (100,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=10)


# Unit test for save_model and load_model function


def test_save_and_load(model):
    filename = 'test_model'
    save_model(model, filename)
    loaded_model = load_model(model, filename)
    assert isinstance(loaded_model, nn.Module)

# Unit test for download_weights function


@patch('advsecurenet.utils.model_utils.requests.get')
@patch('advsecurenet.utils.model_utils.os.path.exists', return_value=True)
def test_download_weights(mock_exists, mock_get):
    weights = download_weights(model_name="resnet50", dataset_name="cifar10")
    # shouldn't download weights if they already exist
    mock_get.assert_not_called()


model_names = ["resnet18", "vgg16", "alexnet",
               "CustomMnistModel", "CustomCifar10Model"]
