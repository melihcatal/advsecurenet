from typing import Optional, Tuple, Union

import torch
from torch.utils.data.distributed import DistributedSampler

from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.shared.types.configs.dataloader_config import \
    DataLoaderConfig
from cli.types.dataloader import DataLoaderCliConfigType


def get_dataloader(config: DataLoaderCliConfigType,
                   train_dataset: Optional[torch.utils.data.Dataset] = None,
                   test_dataset: Optional[torch.utils.data.Dataset] = None,
                   use_ddp: Optional[bool] = False
                   ) -> Union[torch.utils.data.DataLoader, Tuple[torch.utils.data.DataLoader, ...]]:
    """
    Get the dataloader from the configuration.

    Args:
        config (DataLoaderCliConfigType): The dataloader configuration.
        train_dataset (Optional[torch.utils.data.Dataset]): The training dataset.
        test_dataset (Optional[torch.utils.data.Dataset]): The testing dataset.
        use_ddp (Optional[bool]): Whether to use Distributed Data Parallel.

    Returns:
        Union[torch.utils.data.DataLoader, Tuple[torch.utils.data.DataLoader, ...]]:
        Depending on provided datasets, returns the necessary dataloader(s).
    """
    dataloaders = []

    if train_dataset is not None:
        train_loader = DataLoaderFactory.create_dataloader(
            DataLoaderConfig(
                dataset=train_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers_train,
                shuffle=False if use_ddp else config.shuffle_train,
                drop_last=config.drop_last_train,
                pin_memory=config.pin_memory,
                sampler=DistributedSampler(train_dataset) if use_ddp else None
            ))
        dataloaders.append(train_loader)

    if test_dataset is not None:
        test_loader = DataLoaderFactory.create_dataloader(
            DataLoaderConfig(
                dataset=test_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers_test,
                shuffle=config.shuffle_test,
                drop_last=config.drop_last_test,
                pin_memory=config.pin_memory
            ))
        dataloaders.append(test_loader)

    if len(dataloaders) == 1:
        # Return a single DataLoader if only one is provided
        return dataloaders[0]
    # Return a tuple of DataLoaders if both are provided
    return tuple(dataloaders)
