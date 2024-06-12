import pytest
import torch

from advsecurenet.utils.normalization_layer import NormalizationLayer


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_normalization_layer_initialization():
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    norm_layer = NormalizationLayer(mean, std)

    assert torch.equal(norm_layer.mean, torch.tensor(
        mean).view(1, -1, 1, 1).float())
    assert torch.equal(norm_layer.std, torch.tensor(
        std).view(1, -1, 1, 1).float())


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_normalization_layer_forward():
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    norm_layer = NormalizationLayer(mean, std)

    input_tensor = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]],
                                  [[1.0, 1.0], [1.0, 1.0]],
                                  [[1.0, 1.0], [1.0, 1.0]]]])

    expected_output = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]],
                                     [[1.0, 1.0], [1.0, 1.0]],
                                     [[1.0, 1.0], [1.0, 1.0]]]])

    output = norm_layer(input_tensor)

    assert torch.allclose(output, expected_output, atol=1e-6)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_normalization_layer_different_means_and_stds():
    mean = [0.5, 0.5, 0.5]
    std = [0.1, 0.2, 0.3]
    norm_layer = NormalizationLayer(mean, std)

    input_tensor = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]],
                                  [[1.0, 1.0], [1.0, 1.0]],
                                  [[1.0, 1.0], [1.0, 1.0]]]])

    expected_output = torch.tensor([[[[5.0, 5.0], [5.0, 5.0]],
                                     [[2.5, 2.5], [2.5, 2.5]],
                                     [[1.6667, 1.6667], [1.6667, 1.6667]]]])

    output = norm_layer(input_tensor)

    assert torch.allclose(output, expected_output, atol=1e-3)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_normalization_layer_different_device():
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    norm_layer = NormalizationLayer(mean, std)

    input_tensor = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]],
                                  [[1.0, 1.0], [1.0, 1.0]],
                                  [[1.0, 1.0], [1.0, 1.0]]]]).cuda()

    expected_output = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]],
                                     [[1.0, 1.0], [1.0, 1.0]],
                                     [[1.0, 1.0], [1.0, 1.0]]]]).cuda()

    norm_layer = norm_layer.cuda()
    output = norm_layer(input_tensor)

    assert torch.allclose(output, expected_output, atol=1e-6)
    assert output.device == input_tensor.device
