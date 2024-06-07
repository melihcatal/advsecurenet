import os
from typing import List, Optional

import torch


def set_visible_gpus(gpu_ids: Optional[List[int]]) -> None:
    """
    Set the visible GPUs for the current process. If no GPU IDs are provided, all available GPUs are used.

    Args:
        gpu_ids (Optional[List[int]]): The list of GPU IDs to use. Defaults to None. If None, all available GPUs are used.
    """
    if gpu_ids is None:
        try:
            gpu_ids = list(range(torch.cuda.device_count()))
        except Exception:
            print("No GPUs available. Using CPU.")
            return

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)
