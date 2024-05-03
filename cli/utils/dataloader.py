from typing import Tuple

import torch
from torch.utils.data.distributed import DistributedSampler

from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.shared.types.configs.dataloader_config import \
    DataLoaderConfig
from cli.types.dataloader import DataLoaderCliConfigType


def get_dataloader(config: DataLoaderCliConfigType,
                   train_dataset: torch.utils.data.Dataset,
                   test_dataset: torch.utils.data.Dataset,
                   use_ddp: bool = False

                   ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Get the dataloader from the configuration.

    Args:
        config (DataLoaderCliConfigType): The dataloader configuration.
        train_dataset (torch.utils.data.Dataset): The training dataset.
        test_dataset (torch.utils.data.Dataset): The testing dataset.
        use_ddp (bool): Whether to use distributed data parallel.
    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: The training and testing dataloaders.
    """
    train_data_loader = DataLoaderFactory.create_dataloader(
        DataLoaderConfig(
            dataset=train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers_train,
            shuffle=False if use_ddp else config.shuffle_train,
            drop_last=config.drop_last_train,
            pin_memory=config.pin_memory,
            sampler=DistributedSampler(train_dataset) if use_ddp else None
        ))

    test_data_loader = DataLoaderFactory.create_dataloader(
        DataLoaderConfig(
            dataset=test_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers_test,
            shuffle=config.shuffle_test,
            drop_last=config.drop_last_test,
            pin_memory=config.pin_memory
        ))

    return train_data_loader, test_data_loader
