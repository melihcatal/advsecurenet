from typing import Union

from advsecurenet.shared.types.dataset import DatasetType
from advsecurenet.utils.dot_dict import DotDict


class NormalizationParameters:
    """
    Class to handle retrieval and management of normalization parameters for selected datasets.
    The normalization parameters are the mean and standard deviation values for each channel of the dataset.

    Supported datasets:
    - CIFAR-10
    - CIFAR-100
    - ImageNet
    - MNIST
    - SVHN
    - Fashion-MNIST
    """
    DATASETS = {
        DatasetType.CIFAR10: {
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2470, 0.2435, 0.2616]
        },
        DatasetType.CIFAR100: {
            "mean": [0.5071, 0.4867, 0.4408],
            "std": [0.2675, 0.2565, 0.2761]
        },
        DatasetType.IMAGENET: {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        DatasetType.MNIST: {
            "mean": [0.1307],
            "std": [0.3081]
        },
        DatasetType.SVHN: {
            "mean": [0.4377, 0.4438, 0.4728],
            "std": [0.1980, 0.2010, 0.1970]
        },
        DatasetType.FASHION_MNIST: {
            "mean": [0.2860],
            "std": [0.3530]
        }
    }

    @staticmethod
    def get_params(dataset_name: Union[DatasetType, str]
                   ) -> DotDict:
        """
        Retrieve normalization parameters for a specified dataset. The parameters are the mean and standard deviation values for each channel of the dataset.
        Args:
            dataset_name (DatasetType or str): The name of the dataset to retrieve parameters for, either as an enum or string.

        Returns:
            DotDict: A dictionary-like object containing the mean and standard deviation values for the dataset.
        """
        if isinstance(dataset_name, str):
            try:
                # Convert string to DatasetType enum
                dataset_name = DatasetType[dataset_name]
            except KeyError as e:
                raise KeyError(
                    f"Dataset '{dataset_name}' is not supported. Supported datasets are: {NormalizationParameters.list_datasets()}") from e
        params_dict = NormalizationParameters.DATASETS.get(dataset_name)
        return DotDict(params_dict) if params_dict else None

    @staticmethod
    def list_datasets() -> list:
        """
        List available datasets.
        """
        return [key.name for key in NormalizationParameters.DATASETS]
