import random
from typing import Optional

import torch
from torch.utils.data import Subset, TensorDataset, random_split


def get_subset_data(data: torch.utils.data.Dataset, num_samples: int, random_seed: Optional[int] = None) -> torch.utils.data.Dataset:
    """
    Returns a subset of the given dataset.

    Args:
        data (torch.utils.data.Dataset): The dataset to get the subset from.
        num_samples (int): The number of samples to get from the dataset.
        random_seed (Optional[int]): The random seed to use for generating the subset. Defaults to None.

    Returns:
        torch.utils.data.Dataset: The subset of the dataset.
    """
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    indices = random.sample(range(len(data)), num_samples)
    subset = Subset(data, indices)
    return subset


def unnormalize_data(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Unnormalizes the given data using the given mean and standard deviation.

    Args:
        data (torch.Tensor): The data to be unnormalized.
        mean (torch.Tensor): The mean to be used for unnormalization.
        std (torch.Tensor): The standard deviation to be used for unnormalization.

    Returns:
        torch.Tensor: The unnormalized data.
    """
    device = data.device
    return data * torch.tensor(std, device=device).view(1, -1, 1, 1) + torch.tensor(mean, device=device).view(1, -1, 1, 1)


def split_data(x, y, test_size=0.2, val_size=0.25, random_state=None):
    """
    Splits data into train, validation and test sets with the given ratios.

    Args:
        x (list): List of features.
        y (list): List of targets.
        test_size (float): Ratio of test samples.
        val_size (float): Ratio of validation samples.
        random_state (int): Random seed for reproducibility.

    Returns:
        x_train (list): List of training features.
        x_val (list): List of validation features.
        x_test (list): List of test features.
        y_train (list): List of training targets.
        y_val (list): List of validation targets.
        y_test (list): List of test targets.

    """
    # Seed for reproducibility
    if random_state is not None:
        torch.manual_seed(random_state)

    # Convert arrays to torch tensors
    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)
    dataset = TensorDataset(x_tensor, y_tensor)

    # Calculate split sizes
    total_samples = len(x)
    test_samples = int(test_size * total_samples)
    val_samples = int(val_size * (total_samples - test_samples))
    train_samples = total_samples - test_samples - val_samples

    # Perform random splits
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_samples, val_samples, test_samples])

    # Separate features and targets for each split
    x_train = [x[0] for x in train_dataset]
    y_train = [x[1] for x in train_dataset]

    x_val = [x[0] for x in val_dataset]
    y_val = [x[1] for x in val_dataset]

    x_test = [x[0] for x in test_dataset]
    y_test = [x[1] for x in test_dataset]

    return x_train, x_val, x_test, y_train, y_val, y_test
