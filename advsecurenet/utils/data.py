import random
from typing import Optional

import torch
from torch.utils.data import Subset


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
