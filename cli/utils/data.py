
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Subset, random_split

from advsecurenet.datasets.dataset_factory import DatasetFactory
from advsecurenet.shared.types.dataset import DatasetType
from cli.utils.helpers import get_device_from_cfg, to_bchw_format


def load_and_prepare_data(config_data: dict) -> tuple[torch.utils.data.TensorDataset, int, torch.device]:
    """
    Loads and prepares data based on configuration.

    Args:
        config_data (dict): The configuration data.

    Returns:
        tuple[torch.utils.data.TensorDataset, int, torch.device]: A tuple containing the dataset, the number of unique classes in the dataset, and the device to use for the attack.
    """
    device, dataset_type, trained_on_dataset_type = set_device_and_datasets(
        config_data)
    config_data['device'] = device
    config_data['dataset_type'] = dataset_type
    config_data['trained_on_dataset_type'] = trained_on_dataset_type

    data, num_classes = get_data(
        config_data, dataset_type, trained_on_dataset_type)
    return data, num_classes, device


def get_data(config_data, dataset_type, trained_on_dataset_type) -> tuple[torch.utils.data.TensorDataset, int]:
    """
    Fetches and processes data based on configuration.

    Args:
        config_data (dict): The configuration data.
        dataset_type (DatasetType): The type of the dataset to be loaded.
        trained_on_dataset_type (DatasetType): The type of the dataset the model was trained on.

    Returns:
        data (torch.utils.data.TensorDataset): The dataset containing the images and labels.
        num_classes (int): The number of unique classes in the dataset.
    """

    # Initialization
    images, labels = None, None
    num_classes = None

    # Load data based on dataset type
    if dataset_type == DatasetType.CUSTOM:
        images, labels = get_custom_data(config_data['custom_data_dir'])
        trained_on_data_obj = DatasetFactory.create_dataset(
            trained_on_dataset_type)
        num_classes = trained_on_data_obj.num_classes

        data = torch.utils.data.TensorDataset(images, labels)
        return data, num_classes

    dataset_obj = DatasetFactory.create_dataset(dataset_type)
    train_data = dataset_obj.load_dataset(train=True)
    test_data = dataset_obj.load_dataset(train=False)
    all_data = train_data + test_data

    if config_data['dataset_part'] == 'random' and config_data['random_samples'] is None:
        raise ValueError(
            "Please provide a valid number of random samples to use for the attack.")

    if config_data['dataset_part'] == 'random':
        random_samples = min(config_data.get(
            'random_samples', len(all_data)), len(all_data))
        lengths = [random_samples, len(all_data) - random_samples]
        subset, _ = random_split(all_data, lengths)
        random_data = Subset(all_data, subset.indices)

    dataset_map = {
        "train": train_data,
        "test": test_data,
        "all": all_data,
        "random": random_data if config_data['dataset_part'] == 'random' else None
    }

    data = dataset_map.get(config_data['dataset_part'])
    if data is None:
        raise ValueError(
            f"Invalid dataset part specified: {config_data['dataset_part']}")

    num_classes = dataset_obj.num_classes

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
    return data2, num_classes


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
