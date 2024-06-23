import os
import random
import time
from unittest.mock import patch

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
                                                      torch_unseed,
                                                      unique_seed, unseed)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_unique_seed():
    seed1 = unique_seed()
    time.sleep(0.001)  # Sleep for 1 millisecond to ensure a different seed
    seed2 = unique_seed()
    assert seed1 != seed2
    assert 0 <= seed1 < 4294967296
    assert 0 <= seed2 < 4294967296


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_torch_random_seed():
    seed = 42
    torch_random_seed(seed)
    tensor1 = torch.randn(10)
    torch_random_seed(seed)
    tensor2 = torch.randn(10)
    assert torch.equal(tensor1, tensor2)


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('torch.cuda.is_available', return_value=True)
@patch('torch.cuda.manual_seed')
def test_torch_random_seed_cuda(mock_cuda_seed, mock_cuda_available):
    seed = 42
    torch_random_seed(seed)
    mock_cuda_seed.assert_called_once_with(seed)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_numpy_random_seed():
    seed = 42
    numpy_random_seed(seed)
    array1 = np.random.rand(10)
    numpy_random_seed(seed)
    array2 = np.random.rand(10)
    assert np.array_equal(array1, array2)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_python_random_seed():
    seed = 42
    python_random_seed(seed)
    num1 = [random.random() for _ in range(10)]
    python_random_seed(seed)
    num2 = [random.random() for _ in range(10)]
    assert num1 == num2


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_torch_unseed():
    torch_random_seed(42)
    tensor1 = torch.randn(10)
    torch_unseed()
    tensor2 = torch.randn(10)
    assert not torch.equal(tensor1, tensor2)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_numpy_unseed():
    numpy_random_seed(42)
    array1 = np.random.rand(10)
    numpy_unseed()
    array2 = np.random.rand(10)
    assert not np.array_equal(array1, array2)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_python_unseed():
    python_random_seed(42)
    num1 = [random.random() for _ in range(10)]
    python_unseed()
    num2 = [random.random() for _ in range(10)]
    assert num1 != num2


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_unseed():
    torch_random_seed(42)
    numpy_random_seed(42)
    python_random_seed(42)

    tensor1 = torch.randn(10)
    array1 = np.random.rand(10)
    num1 = [random.random() for _ in range(10)]

    unseed()

    tensor2 = torch.randn(10)
    array2 = np.random.rand(10)
    num2 = [random.random() for _ in range(10)]

    assert not torch.equal(tensor1, tensor2)
    assert not np.array_equal(array1, array2)
    assert num1 != num2


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_set_seed():
    set_seed(42)
    tensor1 = torch.randn(10)
    array1 = np.random.rand(10)
    num1 = [random.random() for _ in range(10)]

    set_seed(42)

    tensor2 = torch.randn(10)
    array2 = np.random.rand(10)
    num2 = [random.random() for _ in range(10)]

    assert torch.equal(tensor1, tensor2)
    assert np.array_equal(array1, array2)
    assert num1 == num2


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.utils.reproducibility_utils.set_seed')
@patch('advsecurenet.utils.reproducibility_utils.unseed')
@patch('torch.cuda.is_available', return_value=True)
@patch('torch.backends.cudnn')
def test_set_deterministic(mock_cudnn, mock_is_available, mock_unseed, mock_set_seed):
    set_deterministic(42)

    mock_set_seed.assert_called_once_with(42)
    mock_is_available.assert_called_once()
    assert mock_cudnn.deterministic is True
    assert mock_cudnn.benchmark is False


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.utils.reproducibility_utils.set_seed')
@patch('advsecurenet.utils.reproducibility_utils.unseed')
@patch('torch.cuda.is_available', return_value=False)
@patch('torch.backends.cudnn')
def test_set_deterministic_no_cuda(mock_cudnn, mock_is_available, mock_unseed, mock_set_seed):
    set_deterministic(42)

    mock_set_seed.assert_called_once_with(42)
    mock_is_available.assert_called_once()
    mock_cudnn.deterministic = False  # Ensure these aren't changed
    mock_cudnn.benchmark = True


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.utils.reproducibility_utils.set_seed')
@patch('advsecurenet.utils.reproducibility_utils.unseed')
@patch('torch.cuda.is_available', return_value=True)
@patch('torch.backends.cudnn')
def test_set_nondeterministic(mock_cudnn, mock_is_available, mock_unseed, mock_set_seed):
    set_nondeterministic()

    mock_unseed.assert_called_once()
    mock_is_available.assert_called_once()
    assert mock_cudnn.deterministic is False
    assert mock_cudnn.benchmark is True


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.utils.reproducibility_utils.set_seed')
@patch('advsecurenet.utils.reproducibility_utils.unseed')
@patch('torch.cuda.is_available', return_value=False)
@patch('torch.backends.cudnn')
def test_set_nondeterministic_no_cuda(mock_cudnn, mock_is_available, mock_unseed, mock_set_seed):
    set_nondeterministic()

    mock_unseed.assert_called_once()
    mock_is_available.assert_called_once()
    mock_cudnn.deterministic = True  # Ensure these aren't changed
    mock_cudnn.benchmark = False
