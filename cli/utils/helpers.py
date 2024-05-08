"""
This helpers module contans more general helper functions that are not specific to a command type.
"""
import os
from typing import List, Union

import torch
from torchvision.transforms.functional import to_pil_image


def save_images(images: Union[torch.Tensor, List[torch.Tensor]],
                path: str = None,
                prefix: str = "image") -> None:
    """
    Save each image tensor in a batch or a list of batches to the given path. If no path is provided, the images are saved to the current directory.

    Args:
        images (torch.Tensor or list[torch.Tensor]): A tensor of images or a list of tensors, where each tensor is a batch of images.
        path (str): The path to save the images to. If None, the images are saved to the current directory.
        prefix (str): The prefix to add to the image name.
    """
    # Set the directory where images will be saved
    save_path = path if path else os.getcwd()
    # Make the directory if it does not exist
    os.makedirs(save_path, exist_ok=True)

    # If images input is a single tensor, wrap it in a list
    if isinstance(images, torch.Tensor):
        images = [images]

    # Initialize a counter to uniquely name each image
    image_counter = 0

    # Iterate through each batch in the list
    for batch in images:
        # Check if the tensor is a batch of images
        if len(batch.shape) == 4:
            for img in batch:
                image = to_pil_image(img)  # Convert tensor to PIL image
                save_name = f"{prefix}_{image_counter}.png"
                image.save(os.path.join(save_path, save_name))
                image_counter += 1
        elif len(batch.shape) == 3:
            # Handling the case of a single image in a "batch"
            image = to_pil_image(batch)
            save_name = f"{prefix}_{image_counter}.png"
            image.save(os.path.join(save_path, save_name))
            image_counter += 1


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
