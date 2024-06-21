import logging
import os
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


def set_visible_gpus(gpu_ids: Optional[List[int]] = None) -> None:
    """
    Set the visible GPUs for the current process. If no GPU IDs are provided, all available GPUs are used.

    Args:
        gpu_ids (Optional[List[int]]): The list of GPU IDs to use. Defaults to None. If None, all available GPUs are used.
    """
    if not torch.cuda.is_available():
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        logger.error("CUDA is not available.")
        raise RuntimeError("CUDA is not available.")

    if gpu_ids is None:
        try:
            gpu_ids = list(range(torch.cuda.device_count()))
        except Exception as e:
            logger.error("Error getting CUDA device count. %s" % str(e))
            raise RuntimeError(f"Error getting CUDA device count: {str(e)}")

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)
    logger.info(
        "Set CUDA_VISIBLE_DEVICES to: %s" % os.environ['CUDA_VISIBLE_DEVICES'])
