"""
This module contains the DataLoaderFactory class that creates a DataLoader for the given dataset.
"""

from dataclasses import asdict
from typing import Optional

from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset

from advsecurenet.shared.types.configs.dataloader_config import \
    DataLoaderConfig


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

        # merge the config and kwargs
        params = {**asdict(config), **kwargs}
        dataloader = TorchDataLoader(**params)

        return dataloader
