from typing import Optional

from torchvision import datasets

from advsecurenet.datasets.base_dataset import BaseDataset
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
        self.mean = [0.49139968, 0.48215827, 0.44653124]
        self.std = [0.24703233, 0.24348505, 0.26158768]
        self.input_size = (32, 32)
        self.crop_size = (32, 32)
        self.name = "cifar10"
        self.num_classes = 10
        self.num_input_channels = 3

    def get_dataset_class(self):
        return datasets.CIFAR10
