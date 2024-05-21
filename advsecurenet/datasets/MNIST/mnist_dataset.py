from typing import Optional

import pkg_resources
from torch.utils.data import Dataset as TorchDataset
from torchvision import datasets

from advsecurenet.datasets.base_dataset import BaseDataset, DatasetWrapper
from advsecurenet.shared.types import DataType
from advsecurenet.shared.types.configs.preprocess_config import \
    PreprocessConfig


class MNISTDataset(BaseDataset):
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
        self.mean = [0.1307]
        self.std = [0.3081]
        self.input_size = (28, 28)
        self.crop_size = (28, 28)
        self.name = "mnist"
        self.num_classes = 10

    def load_dataset(self,
                     root: Optional[str] = None,
                     train: Optional[bool] = True,
                     download: Optional[bool] = True,
                     **kwargs) -> DatasetWrapper:
        """
        Loads the MNIST dataset.

        Args:
            root (str, optional): The root directory where the dataset should be stored. Defaults to './data'.
            train (bool, optional): If True, loads the training data. Otherwise, loads the test data. Defaults to True.
            download (bool, optional): If True, downloads the dataset from the internet. Defaults to True.
            **kwargs: Arbitrary keyword arguments for the MNIST dataset.

        Returns:
            DatasetWrapper: The MNIST dataset loaded into memory.
        """

        # If root is not given, use the default data directory
        if root is None:
            root = pkg_resources.resource_filename("advsecurenet", "data")

        transform = self.get_transforms()
        mnist_dataset = datasets.MNIST(
            root=root,
            train=train,
            transform=transform,
            download=download,
            **kwargs)
        self._dataset = DatasetWrapper(
            dataset=mnist_dataset,
            name=self.name,
            **kwargs)

        self.data_type = DataType.TRAIN if train else DataType.TEST
        return self._dataset
