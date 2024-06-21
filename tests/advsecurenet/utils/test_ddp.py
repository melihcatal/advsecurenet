import logging
import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from advsecurenet.utils.ddp import set_visible_gpus


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('torch.cuda.is_available', return_value=True)
@patch('torch.cuda.device_count', return_value=4)
@patch('advsecurenet.utils.ddp.logger')
def test_set_visible_gpus_all(mock_logger, mock_device_count, mock_is_available):
    set_visible_gpus(None)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1,2,3'
    mock_logger.info.assert_called_with('Set CUDA_VISIBLE_DEVICES to: 0,1,2,3')


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('torch.cuda.is_available', return_value=True)
@patch('torch.cuda.device_count', return_value=0)
@patch('advsecurenet.utils.ddp.logger')
def test_set_visible_gpus_no_gpus(mock_logger, mock_device_count, mock_is_available):
    set_visible_gpus(None)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    mock_logger.info.assert_called_with('Set CUDA_VISIBLE_DEVICES to: ')


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('torch.cuda.is_available', return_value=True)
@patch('advsecurenet.utils.ddp.logger')
def test_set_visible_gpus_specific(mock_logger, mock_is_available):
    set_visible_gpus([0, 2])
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,2'
    mock_logger.info.assert_called_with('Set CUDA_VISIBLE_DEVICES to: 0,2')


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('torch.cuda.is_available', return_value=False)
@patch('advsecurenet.utils.ddp.logger')
def test_set_visible_gpus_exception(mock_logger, mock_is_available):
    with pytest.raises(RuntimeError, match="CUDA is not available."):
        set_visible_gpus(None)
    mock_logger.error.assert_called_with(
        'CUDA is not available.')


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('torch.cuda.is_available', return_value=False)
@patch('advsecurenet.utils.ddp.logger')
def test_set_visible_gpus_cuda_not_available(mock_logger, mock_is_available):
    with pytest.raises(RuntimeError, match="CUDA is not available."):
        set_visible_gpus(None)
    assert 'CUDA_VISIBLE_DEVICES' not in os.environ
    mock_logger.error.assert_called_with('CUDA is not available.')
