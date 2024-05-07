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
    def create_dataset(dataset_type: DatasetType, **kwargs) -> BaseDataset:
        """
        Returns a dataset for the given dataset type.

        Parameters
        ----------
        dataset_type: DatasetType
            The type of the dataset to return.
        kwargs: dict
            The keyword arguments to pass to the dataset class.

        Returns
        -------
        BaseDataset
            The dataset for the given dataset type. 

        Raises
        ------
        TypeError
            If the dataset_type is not of type DatasetType.
        """

        if not isinstance(dataset_type, DatasetType):
            raise TypeError(
                f"dataset_type must be of type DatasetType, not {type(dataset_type)}"
            )

        dataset_cls = DATASET_MAP[dataset_type]
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
