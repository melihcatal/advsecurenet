from typing import Union

from advsecurenet.datasets.base_dataset import BaseDataset
from advsecurenet.datasets.Cifar10 import CIFAR10Dataset
from advsecurenet.datasets.ImageNet import ImageNetDataset
from advsecurenet.datasets.MNIST import MNISTDataset
from advsecurenet.shared.types import DatasetType

DATASET_MAP = {
    DatasetType.CIFAR10: CIFAR10Dataset,
    DatasetType.IMAGENET: ImageNetDataset,
    DatasetType.MNIST: MNISTDataset
}


class DatasetFactory:
    """
    A factory class to create datasets.
    """
    @staticmethod
    def create_dataset(dataset_type: DatasetType,
                       return_loaded: bool = False,
                       **kwargs) -> Union[BaseDataset, tuple[BaseDataset, BaseDataset]]:
        """
        Returns a dataset for the given dataset type.

        Args:
            dataset_type (DatasetType): The type of the dataset to be created.
            return_loaded (bool): Whether to load the train and test datasets and return them immediately. Default is False.
            **kwargs: Arbitrary keyword arguments to be passed to the dataset class.

        Returns:
            BaseDataset: The dataset for the given dataset type.

        """

        if not isinstance(dataset_type, DatasetType) and isinstance(dataset_type, str):
            # try to convert the dataset_type to DatasetType
            try:
                dataset_type = DatasetType(dataset_type.upper())
            except ValueError:
                raise TypeError(
                    "dataset_type must be of type DatasetType or a valid string value from DatasetType.")

        dataset_cls = DATASET_MAP[dataset_type]

        if return_loaded:
            dataset_obj = dataset_cls()
            train_dataset = dataset_obj.load_dataset(train=True)
            test_dataset = dataset_obj.load_dataset(train=False)
            return train_dataset, test_dataset

        return dataset_cls(**kwargs)

    @staticmethod
    def available_datasets() -> list:
        """
        Returns a list of available datasets.

        Returns
        -------
        list
            A list of available datasets.
        """

        return list(DATASET_MAP.keys())
