"""
This module contains the DataLoaderFactory class that creates a DataLoader for the given dataset.
"""

import logging
from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset


class DataLoaderFactory:
    """
    The DataLoaderFactory class that creates a DataLoader for the given dataset.

    Attributes: 
        None
    """
    @staticmethod
    def create_dataloader(dataset: TorchDataset, batch_size: int = 32, num_workers: int = 4, shuffle: bool = False, **kwargs) -> TorchDataLoader:
        """
        A static method that creates a DataLoader for the given dataset with the given parameters.

        Args:
            dataset (TorchDataset): The dataset for which the DataLoader is to be created.
            batch_size (int, optional): The batch size. Defaults to 32.
            num_workers (int, optional): The number of workers. Defaults to 4.
            shuffle (bool, optional): If True, shuffles the data. Defaults to False.
            **kwargs: Arbitrary keyword arguments for the DataLoader.

        Returns:
            TorchDataLoader: The DataLoader for the given dataset.

        Raises:
            ValueError: If the dataset is not of type TorchDataset.

        """
        logging.info(f'Creating dataloader for {dataset} dataset')
        if not isinstance(dataset, TorchDataset):
            raise ValueError(
                "Invalid dataset type provided. Expected TorchDataset.")

        dataloader = TorchDataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=num_workers,
                                     **kwargs)

        return dataloader
