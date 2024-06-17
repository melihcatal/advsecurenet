import logging
from typing import Optional

import click
import torch
from torch.utils.data.distributed import DistributedSampler

from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.datasets.base_dataset import BaseDataset
from advsecurenet.shared.types.configs.dataloader_config import \
    DataLoaderConfig
from cli.shared.types.utils.dataloader import DataLoaderCliConfigType

logger = logging.getLogger(__name__)


def get_dataloader(config: DataLoaderCliConfigType,
                   dataset: BaseDataset,
                   dataset_type: Optional[str] = "default",
                   use_ddp: Optional[bool] = False,
                   sampler: Optional[DistributedSampler] = DistributedSampler
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
    if use_ddp and loader_config.shuffle:
        logger.warning(
            "Disabling shuffle for Distributed Data Parallel."
        )
        loader_config.shuffle = False

    sampler = sampler(dataset) if use_ddp else None

    # Create the DataLoader using the factory
    return DataLoaderFactory.create_dataloader(
        DataLoaderConfig(
            dataset=dataset,
            batch_size=loader_config.batch_size,
            num_workers=loader_config.num_workers,
            shuffle=loader_config.shuffle,
            drop_last=loader_config.drop_last,
            pin_memory=loader_config.pin_memory,
            sampler=sampler
        ))
