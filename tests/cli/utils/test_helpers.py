# test_helpers.py
import os

import pytest
import torch
from torchvision.transforms import ToPILImage

from cli.shared.utils.helpers import (get_device_from_cfg, save_img,
                                      to_bchw_format)


class MockConfig:
    def __init__(self, device):
        self.device = device


def test_save_img_custom_path_name(tmp_path):
    tensor = torch.randn(1, 3, 64, 64)  # Example tensor
    custom_path = str(tmp_path)
    custom_name = "test_image.png"
    save_img(tensor, path=custom_path, name=custom_name)
    assert os.path.isfile(os.path.join(custom_path, custom_name))
    # Clean up
    os.remove(os.path.join(custom_path, custom_name))


def test_save_img_default_path_name():
    tensor = torch.randn(1, 3, 64, 64)
    save_img(tensor)
    assert os.path.isfile("image_0.png")
    # Clean up
    os.remove("image_0.png")


def test_save_img_with_batch_dimension(tmp_path):
    tensor = torch.randn(5, 3, 64, 64)  # Batch of 5 images
    save_img(tensor, path=str(tmp_path))
    for i in range(5):
        assert os.path.isfile(os.path.join(tmp_path, f"image_{i}.png"))
        # Clean up
        os.remove(os.path.join(tmp_path, f"image_{i}.png"))


def test_save_img_invalid_input():
    with pytest.raises(AttributeError):
        save_img("not a tensor")


def test_to_bchw_format_conversion():
    tensor = torch.randn(10, 32, 32, 3)
    converted_tensor = to_bchw_format(tensor)
    assert converted_tensor.shape == (10, 3, 32, 32)


def test_to_bchw_format_no_conversion_needed():
    tensor = torch.randn(10, 3, 32, 32)
    converted_tensor = to_bchw_format(tensor)
    assert converted_tensor.shape == tensor.shape


def test_to_bchw_format_invalid_input():
    with pytest.raises(ValueError):
        to_bchw_format(torch.randn(10, 32, 32))


def test_get_device_from_cfg_valid_device():
    config = MockConfig("cuda")
    assert get_device_from_cfg(config) == torch.device("cuda")


def test_get_device_from_cfg_invalid_device():
    config = {"device": "invalid_device"}
    assert get_device_from_cfg(config) == torch.device("cpu")


def test_get_device_from_cfg_dict_config():
    config = {"device": "cuda"}
    assert get_device_from_cfg(config) == torch.device("cuda")


def test_get_device_from_cfg_object_config():
    config = MockConfig("cpu")
    assert get_device_from_cfg(config) == torch.device("cpu")
