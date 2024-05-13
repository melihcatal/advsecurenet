from typing import Optional, Tuple, Union

import torch
from torch.utils.data.distributed import DistributedSampler

from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.shared.types.configs.dataloader_config import \
    DataLoaderConfig
from cli.shared.types.utils.dataloader import DataLoaderCliConfigType


def get_dataloader(config: DataLoaderCliConfigType,
                   dataset_type: Optional[str] = None,
                   dataset: Optional[torch.utils.data.Dataset] = None,
                   use_ddp: Optional[bool] = False
                   ) -> torch.utils.data.DataLoader:
    """
    Get the dataloader based on the provided configuration and dataset type.

    Args:
        config (DataLoaderCliConfigType): The dataloader configuration object.
        dataset_type (Optional[str]): The type of the dataset ('train', 'test', or other).
        dataset (Optional[torch.utils.data.Dataset]): The dataset to be loaded.
        use_ddp (Optional[bool]): Whether to use Distributed Data Parallel.

    Returns:
        torch.utils.data.DataLoader: The configured DataLoader instance.
    """
    if dataset is None:
        raise ValueError("Dataset cannot be None")

    # Determine the configuration to use based on the dataset type
    if dataset_type == 'train' and config.train is not None:
        loader_config = config.train
    elif dataset_type == 'test' and config.test is not None:
        loader_config = config.test
    else:
        loader_config = config.default

    # Adjust for Distributed Data Parallel if needed
    shuffle_setting = loader_config.shuffle and not use_ddp
    sampler = DistributedSampler(dataset) if use_ddp else None

    # Create the DataLoader using the factory
    return DataLoaderFactory.create_dataloader(
        DataLoaderConfig(
            dataset=dataset,
            batch_size=loader_config.batch_size,
            num_workers=loader_config.num_workers,
            shuffle=shuffle_setting,
            drop_last=loader_config.drop_last,
            pin_memory=loader_config.pin_memory,
            sampler=sampler
        ))
