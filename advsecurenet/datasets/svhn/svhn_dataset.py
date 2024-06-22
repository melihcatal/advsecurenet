from typing import Optional

from torchvision import datasets

from advsecurenet.datasets.base_dataset import BaseDataset
from advsecurenet.shared.normalization_params import NormalizationParameters
from advsecurenet.shared.types.configs.preprocess_config import \
    PreprocessConfig


class SVHNDataset(BaseDataset):
    """
    The MNISTDataset class that loads the MNIST dataset.

    Args:
        preprocess_config (Optional[PreprocessConfig], optional): The preprocessing configuration for the MNIST dataset. Defaults to None.

    Attributes:
        mean (List[float]): Mean of the MNIST dataset.
        std (List[float]): Standard deviation of the MNIST dataset.
        input_size (Tuple[int, int]): Input size of the MNIST images.
        name (str): Name of the dataset.
        num_classes (int): Number of classes in the MNIST dataset.
    """

    def __init__(self, preprocess_config: Optional[PreprocessConfig] = None):
        super().__init__(preprocess_config)
        self.mean = NormalizationParameters.get_params("SVHN").mean
        self.std = NormalizationParameters.get_params("SVHN").std
        self.input_size = (32, 32)
        self.crop_size = (32, 32)
        self.name = "svhn"
        self.num_classes = 10
        self.num_input_channels = 3

    def get_dataset_class(self):
        return datasets.SVHN
