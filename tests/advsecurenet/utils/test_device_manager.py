from unittest.mock import MagicMock, patch

import pytest
import torch

from advsecurenet.utils.device_manager import DeviceManager


@pytest.fixture
def device_manager_single():
    return DeviceManager(device="cpu", distributed_mode=False)


@pytest.fixture
def device_manager_distributed():
    return DeviceManager(device="cuda:0", distributed_mode=True)


def test_device_manager_initialization_single(device_manager_single):
    assert device_manager_single.initial_device == "cpu"
    assert not device_manager_single.distributed_mode


def test_device_manager_initialization_distributed(device_manager_distributed):
    assert device_manager_distributed.initial_device == "cuda:0"
    assert device_manager_distributed.distributed_mode


def test_get_current_device_single(device_manager_single):
    current_device = device_manager_single.get_current_device()
    assert current_device == torch.device("cpu")


@patch("torch.cuda.current_device", return_value=1)
def test_get_current_device_distributed(mock_current_device, device_manager_distributed):
    current_device = device_manager_distributed.get_current_device()
    assert current_device == torch.device("cuda:1")
    mock_current_device.assert_called_once()


def test_to_device_single(device_manager_single):
    tensor = torch.randn(3, 3)
    device_tensor = device_manager_single.to_device(tensor)
    assert device_tensor.device == torch.device("cpu")


@patch("torch.cuda.current_device", return_value=0)
def test_to_device_distributed(mock_current_device, device_manager_distributed):
    tensor = torch.randn(3, 3)
    device_tensor = device_manager_distributed.to_device(tensor)
    assert device_tensor.device == torch.device("cuda:0")
    mock_current_device.assert_called_once()


def test_to_device_multiple_tensors_single(device_manager_single):
    tensor1 = torch.randn(3, 3)
    tensor2 = torch.randn(2, 2)
    device_tensors = device_manager_single.to_device(tensor1, tensor2)
    assert device_tensors[0].device == torch.device("cpu")
    assert device_tensors[1].device == torch.device("cpu")


@patch("torch.cuda.current_device", return_value=0)
def test_to_device_multiple_tensors_distributed(mock_current_device, device_manager_distributed):
    tensor1 = torch.randn(3, 3)
    tensor2 = torch.randn(2, 2)
    device_tensors = device_manager_distributed.to_device(tensor1, tensor2)
    assert device_tensors[0].device == torch.device("cuda:0")
    assert device_tensors[1].device == torch.device("cuda:0")
    mock_current_device.assert_called_once()
