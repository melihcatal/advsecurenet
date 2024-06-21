from unittest.mock import patch

import pytest
import torch
from torch import nn

from advsecurenet.models.base_model import BaseModel, check_model_loaded


class MockBaseModel(BaseModel):
    def load_model(self):
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 32 * 32, 10)
        )

    def models(self):
        return ["mock_model"]


@pytest.fixture
def mock_base_model():
    return MockBaseModel()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_forward(mock_base_model):
    x = torch.randn(1, 3, 32, 32)
    output = mock_base_model.forward(x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 10)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_predict(mock_base_model):
    x = torch.randn(1, 3, 32, 32)
    predicted_classes, max_probabilities = mock_base_model.predict(x)
    assert isinstance(predicted_classes, torch.Tensor)
    assert isinstance(max_probabilities, torch.Tensor)
    assert predicted_classes.shape == (1,)
    assert max_probabilities.shape == (1,)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_save_model(mock_base_model, tmp_path):
    path = tmp_path / "model.pth"
    mock_base_model.save_model(path)
    assert path.exists()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_layer_names(mock_base_model):
    with patch('advsecurenet.models.base_model.get_graph_node_names', return_value=([], ["layer1", "layer2"])):
        layer_names = mock_base_model.get_layer_names()
        assert isinstance(layer_names, list)
        assert "layer1" in layer_names
        assert "layer2" in layer_names


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_layer(mock_base_model):
    layer = mock_base_model.get_layer("0")
    assert isinstance(layer, nn.Module)
    assert isinstance(layer, nn.Conv2d)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_set_layer(mock_base_model):
    new_layer = nn.Conv2d(3, 32, kernel_size=3, padding=1)
    mock_base_model.set_layer("0", new_layer)
    assert isinstance(mock_base_model.model[0], nn.Conv2d)
    assert mock_base_model.model[0].out_channels == 32


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_add_layer(mock_base_model):
    new_layer = nn.Conv2d(3, 32, kernel_size=3, padding=1)
    mock_base_model.add_layer(new_layer)
    assert isinstance(mock_base_model.model[-1], nn.Conv2d)
    assert mock_base_model.model[-1].out_channels == 32


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_parent_module_and_name(mock_base_model):
    parent, name = mock_base_model._get_parent_module_and_name("0")
    assert isinstance(parent, nn.Sequential)
    assert name == "0"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_models(mock_base_model):
    models = mock_base_model.models()
    assert isinstance(models, list)
    assert "mock_model" in models

# Mock class to test the decorator


class MockModelClass:
    def __init__(self, model=None):
        self.model = model

    @check_model_loaded
    def some_method(self):
        return "Method called"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_method_with_model_loaded():
    mock_instance = MockModelClass(model="Dummy Model")
    result = mock_instance.some_method()
    assert result == "Method called"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_method_without_model_loaded():
    mock_instance = MockModelClass(model=None)
    with pytest.raises(ValueError, match="Model is not loaded."):
        mock_instance.some_method()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_method_with_custom_model():
    mock_instance = MockModelClass(model="Custom Model")
    result = mock_instance.some_method()
    assert result == "Method called"
