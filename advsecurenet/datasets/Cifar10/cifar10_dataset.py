from typing import Optional

from torchvision import datasets

from advsecurenet.datasets.base_dataset import BaseDataset
from advsecurenet.shared.normalization_params import NormalizationParameters
from advsecurenet.shared.types.configs.preprocess_config import \
    PreprocessConfig


class CIFAR10Dataset(BaseDataset):
    """
    The CIFAR10Dataset class that loads the CIFAR-10 dataset.

    Args:
        preprocess_config (Optional[PreprocessConfig], optional): The preprocessing configuration for the CIFAR-10 dataset. Defaults to None.

    Attributes:
        mean (List[float]): Mean of the CIFAR-10 dataset.
        std (List[float]): Standard deviation of the CIFAR-10 dataset.
        input_size (Tuple[int, int]): Input size of the CIFAR-10 images.
        name (str): Name of the dataset.
        num_classes (int): Number of classes in the CIFAR-10 dataset.
        num_input_channels (int): Number of input channels in the CIFAR-10 images.
    """

    def __init__(self, preprocess_config: Optional[PreprocessConfig] = None):
        super().__init__(preprocess_config)
        self.mean = NormalizationParameters.get_params("CIFAR10").mean
        self.std = NormalizationParameters.get_params("CIFAR10").std
        self.input_size = (32, 32)
        self.crop_size = (32, 32)
        self.name = "cifar10"
        self.num_classes = 10
        self.num_input_channels = 3

    def get_dataset_class(self):
        return datasets.CIFAR10


class CIFAR100Dataset(CIFAR10Dataset):
    """
    The CIFAR100Dataset class that loads the CIFAR-100 dataset.

    Args:
        preprocess_config (Optional[PreprocessConfig], optional): The preprocessing configuration for the CIFAR-100 dataset. Defaults to None.

    Attributes:
        mean (List[float]): Mean of the CIFAR-100 dataset.
        std (List[float]): Standard deviation of the CIFAR-100 dataset.
        input_size (Tuple[int, int]): Input size of the CIFAR-100 images.
        name (str): Name of the dataset.
        num_classes (int): Number of classes in the CIFAR-100 dataset.
        num_input_channels (int): Number of input channels in the CIFAR-100 images.
    """

    def __init__(self, preprocess_config: Optional[PreprocessConfig] = None):
        super().__init__(preprocess_config)
        self.mean = NormalizationParameters.get_params("CIFAR100").mean
        self.std = NormalizationParameters.get_params("CIFAR100").std
        self.name = "cifar100"
        self.num_classes = 100

    def get_dataset_class(self):
        return datasets.CIFAR100
