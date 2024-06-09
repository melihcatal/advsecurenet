"""
This helpers module contans more general helper functions that are not specific to a command type.
"""
import csv
import json
import os
from typing import Any, List, Type, Union

import torch
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm


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

    # Calculate total number of images
    total_images = sum(len(batch) if len(batch.shape)
                       == 4 else 1 for batch in images)

    # Initialize tqdm progress bar with total number of images
    with tqdm(total=total_images, desc="Saving images", unit="image") as pbar:
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
                    pbar.update(1)
            elif len(batch.shape) == 3:
                # Handling the case of a single image in a "batch"
                image = to_pil_image(batch)
                save_name = f"{prefix}_{image_counter}.png"
                image.save(os.path.join(save_path, save_name))
                image_counter += 1
                pbar.update(1)


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


def read_data_from_file(file_path: str, cast_type: Type = str, return_type: Type = list, separator: str = '/n') -> Union[List[Any], set, tuple, torch.Tensor]:
    """
    Reads data from a file and returns it in the specified format. The function supports text, CSV, and JSON files.

    Args:
        file_path (str): The path to the file.
        cast_type (Type): The type to cast the items to. Default is str.
        return_type (Type): The type of collection to return. Default is list. Other options are set and tuple.
        separator (str): The delimiter to use for text and CSV files. Default is '/n' (newline).

    Returns:
        Union[List[Any], set, tuple, torch.Tensor]: The data read from the file in the specified format.
    """
    def read_text_file(file_path: str, cast_type: Type, separator: str) -> List[Any]:
        items = []
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            if separator == '/n':
                for line in content.splitlines():
                    stripped_line = line.strip()
                    if stripped_line:
                        items.append(cast_type(stripped_line))
            else:
                for line in content.splitlines():
                    for item in line.split(separator):
                        stripped_item = item.strip()
                        if stripped_item:
                            items.append(cast_type(stripped_item))
        return items

    def read_csv_file(file_path: str, cast_type: Type, separator: str) -> List[Any]:
        items = []
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter=separator)
            for row in reader:
                for item in row:
                    stripped_item = item.strip()
                    if stripped_item:
                        items.append(cast_type(stripped_item))
        return items

    def read_json_file(file_path: str) -> List[Any]:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    # Determine the file extension
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == '.txt':
        data = read_text_file(file_path, cast_type, separator)
    elif file_extension == '.json':
        data = read_json_file(file_path)
    elif file_extension == '.csv':
        data = read_csv_file(file_path, cast_type, separator)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    # Convert to the desired return type
    if return_type == list:
        return data
    elif return_type == set:
        return set(data)
    elif return_type == tuple:
        return tuple(data)
    elif return_type == torch.Tensor:
        return torch.tensor(data)
    else:
        raise ValueError(f"Unsupported return type: {return_type}")
