from advsecurenet.shared.types.dataset import DatasetType
from advsecurenet.utils.dot_dict import DotDict


class NormalizationParameters:
    """
    Class to handle retrieval and management of normalization parameters for selected datasets.
    The normalization parameters are the mean and standard deviation values for each channel of the dataset.

    Supported datasets:
    - CIFAR-10
    - ImageNet
    - MNIST
    """
    DATASETS = {
        DatasetType.CIFAR10: {
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2470, 0.2435, 0.2616]
        },
        DatasetType.IMAGENET: {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        DatasetType.MNIST: {
            "mean": [0.1307],
            "std": [0.3081]
        }
    }

    @staticmethod
    def get_params(dataset_name):
        """
        Retrieve normalization parameters for a specified dataset. Supported datasets are "CIFAR-10", "ImageNet", and "MNIST".
        Args:
            dataset_name (DatasetType or str): The name of the dataset to retrieve parameters for, either as an enum or string.

        Returns:
            DotDict: A dictionary-like object containing the mean and standard deviation values for the dataset.
        """
        if isinstance(dataset_name, str):
            try:
                # Convert string to DatasetType enum
                dataset_name = DatasetType[dataset_name]
            except KeyError:
                return None  # Return None if the string does not match any DatasetType
        params_dict = NormalizationParameters.DATASETS.get(dataset_name)
        return DotDict(params_dict) if params_dict else None

    @staticmethod
    def list_datasets():
        """
        List available datasets.
        """
        return list(NormalizationParameters.DATASETS.keys())
