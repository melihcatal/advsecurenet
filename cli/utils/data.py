
import os

import click
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Subset, random_split

from advsecurenet.datasets.dataset_factory import DatasetFactory
from advsecurenet.shared.types.dataset import DatasetType
from cli.types.dataset import AttacksDatasetCliConfigType
from cli.utils.helpers import get_device_from_cfg, to_bchw_format


def load_and_prepare_data(config_data: AttacksDatasetCliConfigType) -> torch.utils.data.TensorDataset:
    """
    Loads and prepares data based on configuration.

    Args:
        config_data (AttacksDatasetCliConfigType): The configuration data.

    Returns:
        torch.utils.data.TensorDataset: The dataset containing the images and labels.
    """
    dataset_type = _get_dataset_type(config_data)
    return get_data(config_data, dataset_type)


def _get_dataset_type(config_data: AttacksDatasetCliConfigType) -> DatasetType:
    """
    Returns the dataset type based on the configuration data.

    Args:
        config_data (AttacksDatasetCliConfigType): The configuration data.

    Returns:
        DatasetType: The dataset type.

    """
    dataset_name = config_data.dataset_name.upper()
    if dataset_name not in DatasetType._value2member_map_:
        raise ValueError("Unsupported dataset name! Choose from: " +
                         ", ".join([e.value for e in DatasetType]))

    return DatasetType(dataset_name)


def get_data(config_data: AttacksDatasetCliConfigType,
             dataset_type: DatasetType) -> torch.utils.data.TensorDataset:
    """
    Returns the dataset based on the configuration data and dataset type.

    Args:
        config_data (AttacksDatasetCliConfigType): The configuration data.
        dataset_type (DatasetType): The dataset type.

    Returns:
        torch.utils.data.TensorDataset: The dataset containing the images and labels.

    """

    # Initialization
    images, labels = None, None

    # Load data based on dataset type
    if dataset_type == DatasetType.CUSTOM:
        images, labels = get_custom_data(config_data.custom_data_dir)

        return torch.utils.data.TensorDataset(images, labels)

    dataset_obj = DatasetFactory.create_dataset(dataset_type)
    train_data = dataset_obj.load_dataset(train=True)
    test_data = dataset_obj.load_dataset(train=False)
    all_data = train_data + test_data

    if config_data.dataset_part == 'random' and config_data.random_sample_size is None:
        raise click.UsageError(
            "Please provide the number of random samples to select from the dataset.")

    if config_data.dataset_part == 'random':
        random_samples = min(
            config_data.random_sample_size, len(all_data))
        lengths = [random_samples, len(all_data) - random_samples]
        subset, _ = random_split(all_data, lengths)
        random_data = Subset(all_data, subset.indices)

    dataset_map = {
        "train": train_data,
        "test": test_data,
        "all": all_data,
        "random": random_data if config_data.dataset_part == 'random' else None
    }

    data = dataset_map.get(config_data.dataset_part)
    if data is None:
        raise click.UsageError(
            f"Unsupported dataset part: {config_data.dataset_part}")

    images = [img for img, _ in data]
    labels = [label for _, label in data]

    # convert to tensors
    images = torch.stack([torch.tensor(np.array(image))
                          for image in images]).float()
    # normalize if needed
    if torch.max(images) > 1:
        images /= 255.0

    labels = torch.tensor(labels)
    images = to_bchw_format(images)

    # combine images and labels into a single tensor to have a single data object
    data2 = torch.utils.data.TensorDataset(images, labels)
    return data2


def get_custom_data(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the images and labels from the custom data directory. The expected directory structure is for each class to have its own directory, and the images for that class to be in that directory.


    Args:
        path (str): The path to the custom data directory.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the images and labels.
    """
    images = []
    labels = []

    # Iterate over each subfolder (class)
    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)

        # Ensure that it's a directory and not a file
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                # check if the file is an image
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(class_path, file)
                    image = Image.open(image_path)
                    images.append(image)
                    labels.append(class_name)  # Using folder name as label

    # convert them to tensors
    # image should be tensor of float and shape (batch_size, channels, height, width)
    # Create the tensor
    images_tensor = torch.stack(
        [torch.tensor(np.array(image)) for image in images]).float()

    # Permute the dimensions to [batch, channels, height, width]
    images_tensor = to_bchw_format(images_tensor)
    # normalize the images if needed
    if torch.max(images_tensor) > 1:
        images_tensor /= 255.0

    labels_ids = [labels.index(label) for label in labels]
    labels_tensor = torch.tensor(labels_ids)

    # if we don't have batch dimension, add it
    if len(images_tensor.shape) == 3:
        images_tensor = images_tensor.unsqueeze(0)

    if len(labels_tensor.shape) == 0:
        labels_tensor = labels_tensor.unsqueeze(0)

    return images_tensor, labels_tensor


def set_device_and_datasets(config_data):
    """Sets the device and matches dataset names to dataset types."""

    device = get_device_from_cfg(config_data)

    dataset_name = config_data['dataset_name'].upper()
    if dataset_name not in DatasetType._value2member_map_:
        raise ValueError("Unsupported dataset name! Choose from: " +
                         ", ".join([e.value for e in DatasetType]))

    dataset_type = DatasetType(dataset_name)
    trained_on_dataset_type = DatasetType(config_data['trained_on'].upper())

    return device, dataset_type, trained_on_dataset_type
