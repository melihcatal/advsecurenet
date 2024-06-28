import pytest
import torch
import torch.nn as nn

from advsecurenet.models.CustomModels.CustomCifar10Model import CustomCifar10Model


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_model_initialization():
    # Test model initialization with default parameters
    model = CustomCifar10Model()
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
    model = CustomCifar10Model()
    # Batch of 8, 3 input channels, 32x32 image
    input_tensor = torch.randn(8, 3, 32, 32)
    output = model(input_tensor)
    # Output should have shape (batch_size, num_classes)
    assert output.shape == (8, 10)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_custom_initialization():
    # Test model initialization with custom parameters
    model = CustomCifar10Model(num_classes=20, num_input_channels=1)
    assert model.fc2.out_features == 20
    assert model.conv1.in_channels == 1
