import random

import numpy as np
import pytest
import torch

from advsecurenet.utils.reproducibility_utils import (numpy_random_seed,
                                                      numpy_unseed,
                                                      python_random_seed,
                                                      python_unseed,
                                                      set_deterministic,
                                                      set_nondeterministic,
                                                      set_seed,
                                                      torch_random_seed,
                                                      torch_unseed, unseed)


def test_torch_random_seed():
    torch_random_seed(123)
    value1 = torch.rand(1).item()
    torch_random_seed(123)
    value2 = torch.rand(1).item()
    assert value1 == value2


def test_numpy_random_seed():
    numpy_random_seed(123)
    value1 = np.random.rand()
    numpy_random_seed(123)
    value2 = np.random.rand()
    assert value1 == value2


def test_python_random_seed():
    python_random_seed(123)
    value1 = random.random()
    python_random_seed(123)
    value2 = random.random()
    assert value1 == value2


def test_set_seed():
    set_seed(123)
    value1_torch = torch.rand(1).item()
    value1_np = np.random.rand()
    value1_py = random.random()

    set_seed(123)
    value2_torch = torch.rand(1).item()
    value2_np = np.random.rand()
    value2_py = random.random()

    assert value1_torch == value2_torch
    assert value1_np == value2_np
    assert value1_py == value2_py


def test_set_deterministic():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    set_deterministic(123)
    assert torch.backends.cudnn.deterministic == True
    assert not torch.backends.cudnn.benchmark


def test_set_nondeterministic():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    set_nondeterministic()
    assert not torch.backends.cudnn.deterministic
    assert torch.backends.cudnn.benchmark


def test_torch_unseed():
    torch_unseed()
    value1 = torch.rand(1).item()
    torch_unseed()
    value2 = torch.rand(1).item()
    assert value1 != value2


def test_numpy_unseed():
    numpy_unseed()
    value1 = np.random.rand()
    numpy_unseed()
    value2 = np.random.rand()
    assert value1 != value2


def test_python_unseed():
    python_unseed()
    value1 = random.random()
    python_unseed()
    value2 = random.random()
    assert value1 != value2


def test_unseed():
    set_seed(123)
    unseed()
    value1_torch = torch.rand(1).item()
    value1_np = np.random.rand()
    value1_py = random.random()

    set_seed(123)
    value2_torch = torch.rand(1).item()
    value2_np = np.random.rand()
    value2_py = random.random()

    assert value1_torch != value2_torch
    assert value1_np != value2_np
    assert value1_py != value2_py
