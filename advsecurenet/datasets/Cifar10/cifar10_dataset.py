from typing import Optional
from torchvision import datasets
from torch.utils.data import Dataset as TorchDataset
from advsecurenet.datasets.base_dataset import BaseDataset, DatasetWrapper
from advsecurenet.shared.types import DataType
import pkg_resources


class CIFAR10Dataset(BaseDataset):
    """
    The CIFAR10Dataset class that loads the CIFAR-10 dataset.

    Attributes:
        mean (List[float]): Mean of the CIFAR-10 dataset.
        std (List[float]): Standard deviation of the CIFAR-10 dataset.
        input_size (Tuple[int, int]): Input size of the CIFAR-10 images.
        name (str): Name of the dataset.
        num_classes (int): Number of classes in the CIFAR-10 dataset.
        num_input_channels (int): Number of input channels in the CIFAR-10 images.
    """

    def __init__(self):
        super().__init__()
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.input_size = (32, 32)
        self.name = "cifar10"
        self.num_classes = 10
        self.num_input_channels = 3

    def load_dataset(self,
                     root: Optional[str] = None,
                     train: Optional[bool] = True,
                     download: Optional[bool] = True,
                     **kwargs) -> DatasetWrapper:
        """
        Loads the CIFAR-10 dataset.

        Args:
            root (str, optional): The root directory where the dataset should be stored. Defaults to './data'.
            train (bool, optional): If True, loads the training data. Otherwise, loads the test data. Defaults to True.
            download (bool, optional): If True, downloads the dataset from the internet. Defaults to True.
            **kwargs: Arbitrary keyword arguments for the CIFAR-10 dataset.

        Returns:
            DatasetWrapper: The CIFAR-10 dataset loaded into memory.
        """

        # If root is not given, use the default data directory
        if root is None:
            root = pkg_resources.resource_filename("advsecurenet", "data")

        transform = self.get_transforms()
        cifar10_dataset = datasets.CIFAR10(
            root=root,
            train=train,
            transform=transform,
            download=download, **kwargs)
        self._dataset = DatasetWrapper(
            dataset=cifar10_dataset,
            name=self.name)
        self.data_type = DataType.TRAIN if train else DataType.TEST
        return self._dataset
