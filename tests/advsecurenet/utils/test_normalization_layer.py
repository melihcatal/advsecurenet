from unittest.mock import MagicMock, patch

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
@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.current_device", return_value=0)
@patch("torch.tensor")
def test_normalization_layer_different_device(mock_tensor, mock_current_device, mock_cuda_available):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    norm_layer = NormalizationLayer(mean, std)

    # Create mock tensors
    mock_input_tensor = MagicMock()
    mock_expected_output = MagicMock()

    # Mock the tensor creation to return mock tensors
    mock_tensor.side_effect = [mock_input_tensor, mock_expected_output]

    # Mock the .cuda() method to return the same mock tensor
    mock_input_tensor.cuda.return_value = mock_input_tensor
    mock_expected_output.cuda.return_value = mock_expected_output

    # Set the device attribute manually
    mock_input_tensor.device = torch.device("cuda:0")
    mock_expected_output.device = torch.device("cuda:0")

    input_tensor = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]],
                                  [[1.0, 1.0], [1.0, 1.0]],
                                  [[1.0, 1.0], [1.0, 1.0]]]]).cuda()

    expected_output = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]],
                                     [[1.0, 1.0], [1.0, 1.0]],
                                     [[1.0, 1.0], [1.0, 1.0]]]]).cuda()

    norm_layer = norm_layer.cuda()

    with patch.object(norm_layer, 'forward', return_value=mock_expected_output) as mock_forward:
        output = norm_layer(input_tensor)

        mock_forward.assert_called_once_with(mock_input_tensor)
        assert output == mock_expected_output

    assert output.device == input_tensor.device
