import random
import torch
from typing import Optional
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
