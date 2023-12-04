import os
import random
import time

import numpy as np
import torch


def unique_seed() -> int:
    """
    Generates a unique seed for each run. This combines the current time and the process ID.
    """

    current_time = int(time.time() * 1000000)  # Microseconds
    pid = os.getpid()  # Process ID
    seed = current_time ^ pid  # XOR to combine the values
    # make sure that the seed is between 0 and 2^32 - 1
    seed = seed % 4294967296
    return seed


def torch_random_seed(seed: int = 42) -> None:
    """
    Sets the random seed for PyTorch and its related libraries.

    Parameters
    ----------
    seed : int, optional
        The random seed to use, by default 42

    """
    torch.manual_seed(seed)
    # Check if CUDA is available and set the random seed for GPU as well.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def numpy_random_seed(seed: int = 42) -> None:
    """
    Sets the random seed for NumPy and its related libraries.

    Parameters
    ----------
    seed : int, optional
        The random seed to use, by default 42

    """
    np.random.seed(seed)


def python_random_seed(seed: int = 42) -> None:
    """
    Sets the random seed for Python's random module.

    Parameters
    ----------
    seed : int, optional
        The random seed to use, by default 42

    """
    random.seed(seed)


def torch_unseed() -> None:
    """
    Unsets the random seed for PyTorch and its related libraries.

    """
    seed = unique_seed()
    torch_random_seed(seed)


def numpy_unseed() -> None:
    """
    Unsets the random seed for NumPy and its related libraries.

    """
    seed = unique_seed()
    numpy_random_seed(seed)


def python_unseed() -> None:
    """
    Unsets the random seed for Python's random module.

    """
    seed = unique_seed()
    python_random_seed(seed)


def unseed() -> None:
    """
    Unsets the random seed for all libraries.

    """
    torch_unseed()
    numpy_unseed()
    python_unseed()


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for all libraries.

    Parameters
    ----------
    seed : int, optional
        The random seed to use, by default 42

    """
    torch_random_seed(seed)
    numpy_random_seed(seed)
    python_random_seed(seed)


def set_deterministic(seed: int = 42) -> None:
    """
    Sets the random seed for all libraries and sets the deterministic flag for CUDA.

    Args:
        seed (int, optional): The random seed to use. Defaults to 42.

    """
    set_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_nondeterministic() -> None:
    """
    Unsets the random seed for all libraries and unsets the deterministic flag for CUDA.

    """
    unseed()
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


__all__ = [
    "set_seed",
    "unseed",
    "set_deterministic",
    "set_nondeterministic",
]
