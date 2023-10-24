import pytest
from torch import nn
import torch
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader, TensorDataset
from advsecurenet.utils.model_utils import train, save_model, load_model, download_weights, test as evaluate_model


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


# Unit test for train function
def test_train(model, dataloader):
    with patch('advsecurenet.utils.model_utils.tqdm', return_value=dataloader) as mock_tqdm:
        train(model, dataloader, epochs=1) 

# Unit test for test function
def test_evaluate(model, dataloader):
    with patch('advsecurenet.utils.model_utils.tqdm', return_value=dataloader) as mock_tqdm:
        test_loss, accuracy = evaluate_model(model, dataloader)
    assert test_loss is not None
    assert accuracy is not None

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
    


