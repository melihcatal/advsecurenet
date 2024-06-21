from typing import Optional, Union

from advsecurenet.datasets.base_dataset import BaseDataset
from advsecurenet.datasets.Cifar10 import CIFAR10Dataset, CIFAR100Dataset
from advsecurenet.datasets.Custom import CustomDataset
from advsecurenet.datasets.ImageNet import ImageNetDataset
from advsecurenet.datasets.MNIST import FashionMNISTDataset, MNISTDataset
from advsecurenet.datasets.svhn import SVHNDataset
from advsecurenet.shared.types import DatasetType
from advsecurenet.shared.types.configs.preprocess_config import \
    PreprocessConfig

DATASET_MAP = {
    DatasetType.CIFAR10: CIFAR10Dataset,
    DatasetType.IMAGENET: ImageNetDataset,
    DatasetType.MNIST: MNISTDataset,
    DatasetType.FASHION_MNIST: FashionMNISTDataset,
    DatasetType.CIFAR100: CIFAR100Dataset,
    DatasetType.SVHN: SVHNDataset,

    DatasetType.CUSTOM: CustomDataset,

}


class DatasetFactory:
    """
    A factory class to create datasets.
    """
    @staticmethod
    def create_dataset(dataset_type: Union[DatasetType, str],
                       preprocess_config: Optional[PreprocessConfig] = None,
                       return_loaded: Optional[bool] = False,
                       **kwargs) -> Union[BaseDataset, tuple[BaseDataset, BaseDataset]]:
        """
        Returns a dataset for the given dataset type.

        Args:
            dataset_type (Union[DatasetType, str]): The dataset type to create.
            return_loaded (bool): Whether to load the train and test datasets and return them immediately. Default is False.
            **kwargs: Arbitrary keyword arguments to be passed to the dataset class.

        Returns:
            BaseDataset: The dataset for the given dataset type.

        """

        if not isinstance(dataset_type, DatasetType) and isinstance(dataset_type, str):
            # try to convert the dataset_type to DatasetType
            try:
                dataset_type = DatasetType(dataset_type.upper())
            except ValueError as exc:
                raise TypeError(
                    'dataset_type must be of type DatasetType or a valid string value from DatasetType.') from exc

        dataset_cls = DATASET_MAP[dataset_type]

        if return_loaded:
            dataset_obj = dataset_cls(preprocess_config)
            train_dataset = dataset_obj.load_dataset(train=True)
            test_dataset = dataset_obj.load_dataset(train=False)
            return train_dataset, test_dataset

        return dataset_cls(preprocess_config, **kwargs)

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
