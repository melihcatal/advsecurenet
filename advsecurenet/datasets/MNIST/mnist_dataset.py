from advsecurenet.datasets.base_dataset import BaseDataset
from advsecurenet.shared.types import DataType
from torchvision import datasets
from torch.utils.data import Dataset as TorchDataset
from typing import Optional


class MNISTDataset(BaseDataset):
    """
    The MNISTDataset class that loads the MNIST dataset.

    Attributes:
        mean (List[float]): Mean of the MNIST dataset.
        std (List[float]): Standard deviation of the MNIST dataset.
        input_size (Tuple[int, int]): Input size of the MNIST images.
        name (str): Name of the dataset.
        num_classes (int): Number of classes in the MNIST dataset.
    """

    def __init__(self):
        super().__init__()
        self.mean = [0.1307]  # mean of MNIST
        self.std = [0.3081]   # std of MNIST
        self.input_size = (28, 28)
        self.name = "mnist"
        self.num_classes = 10

    def load_dataset(self,
                     root: Optional[str] = './data',
                     train: Optional[bool] = True,
                     download: Optional[bool] = True,
                     **kwargs) -> TorchDataset:
        """
        Loads the MNIST dataset.

        Args:
            root (str, optional): The root directory where the dataset should be stored. Defaults to './data'.
            train (bool, optional): If True, loads the training data. Otherwise, loads the test data. Defaults to True.
            download (bool, optional): If True, downloads the dataset from the internet. Defaults to True.
            **kwargs: Arbitrary keyword arguments for the MNIST dataset.

        Returns:
            TorchDataset: The MNIST dataset loaded into memory.
        """
        transform = self.get_transforms()
        self._dataset = datasets.MNIST(
            root=root, train=train, transform=transform, download=download, **kwargs)

        self.data_type = DataType.TRAIN if train else DataType.TEST
        return self._dataset
