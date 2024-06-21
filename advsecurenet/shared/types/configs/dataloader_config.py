from dataclasses import dataclass
from typing import Optional

from torch.utils.data.distributed import DistributedSampler

from advsecurenet.datasets.base_dataset import BaseDataset


@dataclass
class DataLoaderConfig:
    """
    This dataclass is used to store the configuration of the data loader.
    """
    dataset: BaseDataset
    batch_size: Optional[int] = 16
    num_workers: Optional[int] = 4
    shuffle: Optional[bool] = True
    sampler: Optional[DistributedSampler] = None
    pin_memory: Optional[bool] = True
    drop_last: Optional[bool] = False
