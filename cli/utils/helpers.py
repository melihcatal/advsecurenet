"""
This helpers module contans more general helper functions that are not specific to a command type.
"""
import os

import torch
from torchvision.transforms import ToPILImage


def save_img(img: torch.tensor, path: str = None, name: str = None) -> None:
    """
    Save an image tensor to the given path. If no path is provided, the image is saved to the current directory.

    Args:

    img (torch.tensor): The image tensor to save.
    path (str): The path to save the image to. If None, the image is saved to the current directory.
    name (str): The name of the image. If None, the image is saved as 'image_{i}.png' where i is the index of the image tensor in the batch.
    """

    to_pil = ToPILImage()

    # squeeze the batch dimension if it exists
    if len(img.shape) == 4:
        img = img.squeeze(0)

    save_path = path if path else os.getcwd()
    # Save the images to the provided path. If no path is provided, save to the current directory.
    for i, image_tensor in enumerate(img):
        image = to_pil(image_tensor)  # Convert tensor to PIL image
        save_name = name if name else f"image_{i}.png"
        image.save(os.path.join(save_path, save_name))


def to_bchw_format(tensor):
    """
    Converts a tensor from BHWC (batch, height, width, channels) format to BCHW (batch, channels, height, width) format. 

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor in BCHW format.

    Raises:
        ValueError: If the tensor dimensions do not match expected BHWC or BCHW formats for RGB or grayscale images.

    Examples:

        >>> tensor = torch.randn(10, 32, 32, 3) 
        >>> tensor.shape
        torch.Size([10, 32, 32, 3])
        >>> tensor = to_bchw_format(tensor)
        >>> tensor.shape
        torch.Size([10, 3, 32, 32])
    """
    # Check if the tensor is already in BCHW format
    if len(tensor.shape) == 4 and (tensor.shape[1] == 3 or tensor.shape[1] == 1):
        return tensor
    # Check if the tensor is in BHWC format (either RGB or grayscale) and convert to BCHW
    if len(tensor.shape) == 4 and (tensor.shape[3] == 3 or tensor.shape[3] == 1):
        return tensor.permute(0, 3, 1, 2)
    raise ValueError(
        "Tensor dimensions do not match expected BHWC or BCHW formats for RGB or grayscale images")


def get_device_from_cfg(config) -> torch.device:
    """
    Returns the device to use from the config. If the device is not specified or is invalid, the device is set to "cpu".

    Args:
        config (any): The config object to use.
    """
    try:
        # First, try attribute-style access
        device_str = config.device
    except AttributeError:
        try:
            # If attribute-style access fails, try dictionary-style access
            device_str = config["device"]
        except (KeyError, TypeError):
            # If both attempts fail, default to 'cpu'
            device_str = "cpu"

    # Try to create a torch.device with the obtained string
    try:
        device = torch.device(device_str)
    except Exception:
        device = torch.device("cpu")

    return device
