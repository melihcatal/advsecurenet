import pytest
from unittest.mock import patch
from advsecurenet.utils import get_device
from advsecurenet.shared.types import DeviceType

class MockDeviceProperties:
    def __init__(self, is_multiprocessor):
        self.is_multiprocessor = is_multiprocessor

def test_get_device_cuda_multiprocessor():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=MockDeviceProperties(True)), \
         patch("torch.backends.mps.is_available", return_value=False):
        assert get_device() == DeviceType.CUDA_0

def test_get_device_cuda():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=MockDeviceProperties(False)), \
         patch("torch.backends.mps.is_available", return_value=False):
        assert get_device() == DeviceType.CUDA


def test_get_device_mps():
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=True), \
         patch("torch.backends.mps.is_built", return_value=True):
        assert get_device() == DeviceType.MPS

def test_get_device_cpu():
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=False):
        assert get_device() == DeviceType.CPU
