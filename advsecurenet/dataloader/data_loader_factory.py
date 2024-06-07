"""
This module contains the DataLoaderFactory class that creates a DataLoader for the given dataset.
"""

from dataclasses import asdict, is_dataclass
from typing import Optional

from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler

from advsecurenet.shared.types.configs.dataloader_config import \
    DataLoaderConfig


def dataclass_to_dict(instance):
    """
    Convert a dataclass instance to a dictionary, handling non-serializable fields appropriately.
    """
    result = {}
    for field in instance.__dataclass_fields__:
        value = getattr(instance, field)
        # Handle special cases for non-serializable fields if necessary
        if isinstance(value, (DistributedSampler, TorchDataset)):
            result[field] = value
        elif is_dataclass(value):
            result[field] = asdict(value)
        else:
            result[field] = value
    return result


class DataLoaderFactory:
    """
    The DataLoaderFactory class that creates a DataLoader for the given dataset.

    Attributes: 
        None
    """
    @staticmethod
    def create_dataloader(config: Optional[DataLoaderConfig] = None,
                          **kwargs) -> TorchDataLoader:
        """
        A static method that creates a DataLoader for the given dataset with the given parameters.

        Args:
            config (DataLoaderConfig): The DataLoader configuration.
            **kwargs: Arbitrary keyword arguments for the DataLoader.

        Returns:
            TorchDataLoader: The DataLoader for the given dataset.

        Raises:
            ValueError: If the dataset is not of type TorchDataset.

        Note:
            It is possible to create a DataLoader without providing a DataLoaderConfig. In this case, the DataLoader will be created with the provided keyword arguments.
            DataLoaderConfig contains the following fields:
                - dataset: TorchDataset
                - batch_size: int
                - num_workers: int
                - shuffle: bool
                - drop_last: bool
                - pin_memory: bool
                - sampler: Optional[torch.utils.data.Sampler]

        """
        if config is None:
            # if no config is provided then create a new config based on the kwargs
            config = DataLoaderConfig(**kwargs)

        if not isinstance(config.dataset, TorchDataset):
            raise ValueError(
                "Invalid dataset type provided. Expected TorchDataset.")

        if config.sampler is not None and config.shuffle:
            config.shuffle = False

        config_dict = dataclass_to_dict(config)
        # merge the config and kwargs
        params = {**config_dict, **kwargs}
        dataloader = TorchDataLoader(**params)

        return dataloader
