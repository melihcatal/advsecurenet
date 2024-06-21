import pytest
import torch
import torch.nn as nn

from advsecurenet.models.CustomModels.CustomMnistModel import CustomMnistModel


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_model_initialization():
    # Test model initialization with default parameters
    model = CustomMnistModel()
    assert isinstance(model, nn.Module)
    assert isinstance(model.conv1, nn.Conv2d)
    assert isinstance(model.conv2, nn.Conv2d)
    assert isinstance(model.fc1, nn.Linear)
    assert isinstance(model.fc2, nn.Linear)
    assert isinstance(model.relu, nn.ReLU)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_forward_pass():
    # Test forward pass with default parameters
    model = CustomMnistModel()
    # Batch of 8, 1 input channel, 28x28 image
    input_tensor = torch.randn(8, 1, 28, 28)
    output = model(input_tensor)
    # Output should have shape (batch_size, num_classes)
    assert output.shape == (8, 10)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_custom_initialization():
    # Test model initialization with custom parameters
    model = CustomMnistModel(num_classes=20, num_input_channels=3)
    assert model.fc2.out_features == 20
    assert model.conv1.in_channels == 3
