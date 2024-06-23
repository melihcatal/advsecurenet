import os
from unittest.mock import MagicMock, mock_open, patch

import pytest
import torch

from cli.shared.utils.helpers import (get_device_from_cfg, read_data_from_file,
                                      save_images, to_bchw_format)


@pytest.mark.cli
@pytest.mark.essential
@patch("os.makedirs")
@patch("cli.shared.utils.helpers.to_pil_image", autospec=True)
@patch("builtins.open", new_callable=mock_open)
@patch("os.path.join", side_effect=lambda *args: "/".join(args))
@patch("tqdm.auto.tqdm")
def test_save_images(mock_tqdm, mock_path_join, mock_open, mock_to_pil_image, mock_makedirs):
    mock_image = MagicMock()
    mock_to_pil_image.return_value = mock_image

    images = torch.randn(2, 3, 32, 32)  # 2 images in batch
    save_images(images, path="test_path", prefix="test_prefix")

    mock_makedirs.assert_called_once_with("test_path", exist_ok=True)
    assert mock_to_pil_image.call_count == 2
    assert mock_image.save.call_count == 2

    # Verify that the images were saved correctly
    expected_calls = [
        ("test_path/test_prefix_0.png",),
        ("test_path/test_prefix_1.png",)
    ]
    actual_calls = [call[0] for call in mock_image.save.call_args_list]
    assert actual_calls == expected_calls


@pytest.mark.cli
@pytest.mark.essential
@patch("os.makedirs")
@patch("cli.shared.utils.helpers.to_pil_image", autospec=True)
@patch("builtins.open", new_callable=mock_open)
@patch("os.path.join", side_effect=lambda *args: "/".join(args))
@patch("tqdm.auto.tqdm")
def test_save_images_no_path(mock_tqdm, mock_path_join, mock_open, mock_to_pil_image, mock_makedirs):
    mock_image = MagicMock()
    mock_to_pil_image.return_value = mock_image

    images = torch.randn(1, 3, 32, 32)  # 1 image in batch
    save_images(images, path=None, prefix="test_prefix")

    mock_makedirs.assert_called_once_with(os.getcwd(), exist_ok=True)
    assert mock_to_pil_image.call_count == 1
    assert mock_image.save.call_count == 1

    # Verify that the image was saved correctly
    expected_path = os.path.join(os.getcwd(), "test_prefix_0.png")
    expected_calls = [(expected_path,)]
    actual_calls = [call[0] for call in mock_image.save.call_args_list]
    assert actual_calls == expected_calls


@pytest.mark.cli
@pytest.mark.essential
@patch("os.makedirs")
@patch("cli.shared.utils.helpers.to_pil_image", autospec=True)
@patch("builtins.open", new_callable=mock_open)
@patch("os.path.join", side_effect=lambda *args: "/".join(args))
@patch("tqdm.auto.tqdm")
def test_save_single_image(mock_tqdm, mock_path_join, mock_open, mock_to_pil_image, mock_makedirs):
    mock_image = MagicMock()
    mock_to_pil_image.return_value = mock_image

    images = torch.randn(3, 32, 32)  # Single image
    save_images(images, path="test_path", prefix="test_prefix")

    mock_makedirs.assert_called_once_with("test_path", exist_ok=True)
    assert mock_to_pil_image.call_count == 1
    assert mock_image.save.call_count == 1

    # Verify that the image was saved correctly
    expected_calls = [("test_path/test_prefix_0.png",)]
    actual_calls = [call[0] for call in mock_image.save.call_args_list]
    assert actual_calls == expected_calls


@pytest.mark.cli
@pytest.mark.essential
def test_to_bchw_format_bhwc():
    tensor = torch.randn(10, 32, 32, 3)  # BHWC format
    result = to_bchw_format(tensor)
    assert result.shape == (10, 3, 32, 32)


@pytest.mark.cli
@pytest.mark.essential
def test_to_bchw_format_bchw():
    tensor = torch.randn(10, 3, 32, 32)  # BCHW format
    result = to_bchw_format(tensor)
    assert result.shape == (10, 3, 32, 32)


@pytest.mark.cli
@pytest.mark.essential
def test_to_bchw_format_invalid():
    tensor = torch.randn(10, 32, 32)  # Invalid format
    with pytest.raises(ValueError):
        to_bchw_format(tensor)


@pytest.mark.cli
@pytest.mark.essential
def test_get_device_from_cfg_attr():
    config = MagicMock()
    config.device = "cuda"
    device = get_device_from_cfg(config)
    assert device.type == "cuda"


@pytest.mark.cli
@pytest.mark.essential
def test_get_device_from_cfg_dict():
    config = {"device": "cuda"}
    device = get_device_from_cfg(config)
    assert device.type == "cuda"


@pytest.mark.cli
@pytest.mark.essential
def test_get_device_from_cfg_invalid():
    config = MagicMock()
    del config.device  # Simulate missing attribute
    config.__getitem__.side_effect = KeyError
    device = get_device_from_cfg(config)
    assert device.type == "cpu"


@pytest.mark.cli
@pytest.mark.essential
@patch("torch.cuda.is_available", return_value=False)
def test_get_device_from_cfg_exception(mock_is_available):
    config = MagicMock()
    config.device = "cuda"
    with pytest.raises(Exception):
        device = get_device_from_cfg(config)
        assert device.type == "cpu"


@pytest.mark.cli
@pytest.mark.essential
@patch("builtins.open", new_callable=mock_open, read_data="line1\nline2\n")
def test_read_data_from_file_text(mock_open):
    data = read_data_from_file(
        "test.txt", cast_type=str, return_type=list, separator='\n')
    assert data == ["line1", "line2"]


@pytest.mark.cli
@pytest.mark.essential
@patch("builtins.open", new_callable=mock_open, read_data="item1,item2\nitem3,item4\n")
def test_read_data_from_file_csv(mock_open):
    data = read_data_from_file(
        "test.csv", cast_type=str, return_type=list, separator=',')
    assert data == ["item1", "item2", "item3", "item4"]


@pytest.mark.cli
@pytest.mark.essential
@patch("builtins.open", new_callable=mock_open, read_data='["item1", "item2", "item3"]')
def test_read_data_from_file_json(mock_open):
    data = read_data_from_file("test.json", cast_type=str, return_type=list)
    assert data == ["item1", "item2", "item3"]


@pytest.mark.cli
@pytest.mark.essential
def test_read_data_from_file_invalid_extension():
    with pytest.raises(ValueError):
        read_data_from_file("test.invalid", cast_type=str, return_type=list)


@pytest.mark.cli
@pytest.mark.essential
@patch("builtins.open", new_callable=mock_open, read_data="line1\nline2\n")
def test_read_data_from_file_text_set(mock_open):
    data = read_data_from_file(
        "test.txt", cast_type=str, return_type=set, separator='\n')
    assert data == {"line1", "line2"}


@pytest.mark.cli
@pytest.mark.essential
@patch("builtins.open", new_callable=mock_open, read_data="line1\nline2\n")
def test_read_data_from_file_text_tuple(mock_open):
    data = read_data_from_file(
        "test.txt", cast_type=str, return_type=tuple, separator='\n')
    assert data == ("line1", "line2")


@pytest.mark.cli
@pytest.mark.essential
@patch("builtins.open", new_callable=mock_open, read_data="line1\nline2\n")
def test_read_data_from_file_text_tensor_str(mock_open):
    # pytorch tensor with string values is not supported
    with pytest.raises(ValueError):
        read_data_from_file("test.txt", cast_type=str,
                            return_type=torch.Tensor, separator='\n')


@pytest.mark.cli
@pytest.mark.essential
@patch("builtins.open", new_callable=mock_open, read_data="1\n2\n")
def test_read_data_from_file_text_tensor_int(mock_open):
    data = read_data_from_file(
        "test.txt", cast_type=int, return_type=torch.Tensor, separator='\n')
    assert torch.equal(data, torch.tensor([1, 2]))


@pytest.mark.cli
@pytest.mark.essential
@patch("builtins.open", new_callable=mock_open, read_data="line1\nline2\n")
def test_read_data_from_file_text_invalid_return_type(mock_open):
    with pytest.raises(ValueError) as exc_info:
        read_data_from_file("test.txt", cast_type=str,
                            return_type=int, separator='\n')

    assert "Unsupported return type" in str(exc_info.value)
