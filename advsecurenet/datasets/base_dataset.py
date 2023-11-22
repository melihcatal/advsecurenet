from torchvision import datasets, transforms
from torch.utils.data import Dataset as TorchDataset
from typing import Tuple, Any, List
from abc import ABC, abstractmethod
from advsecurenet.shared.types.dataset import DataType
from typing import Optional


class DatasetWrapper(TorchDataset):
    def __init__(self, dataset, name):
        self.dataset = dataset
        self.name = name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class BaseDataset(TorchDataset, ABC):
    """
    A base class for PyTorch datasets.

    Attributes:
        _dataset (TorchDataset): The underlying PyTorch dataset.
        mean (List[float]): The mean values of the dataset.
        std (List[float]): The standard deviation values of the dataset.
        input_size (Tuple[int, int]): The input size of the dataset.
        name (str): The name of the dataset.
        num_classes (int): The number of classes in the dataset.
        num_input_channels (int): The number of input channels in the dataset.
        data_type (DataType): The type of the dataset (train or test).
    """

    def __init__(self):
        self._dataset: DatasetWrapper
        self.mean: List[float] = []
        self.std: List[float] = []
        self.input_size: Tuple[int, int] = ()
        self.name: str = ""
        self.num_classes: int = 0
        self.num_input_channels: int = 0
        self.data_type: DataType = None

    def get_transforms(self) -> transforms.Compose:
        """
        Returns the data transforms to be applied to the dataset.

        Returns:
            transforms.Compose: The data transforms.
        """
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        if self._dataset:
            return len(self._dataset)
        return 0

    def __getitem__(self, idx: int) -> Any:
        """
        Returns the item at the given index.

        Args:
            idx (int): The index of the item.

        Returns:
            Any: The item at the given index.

        Raises:
            NotImplementedError: If the dataset is not loaded or specified.
        """
        if self._dataset:
            return self._dataset[idx]
        raise NotImplementedError("Dataset not loaded or specified.")

    @abstractmethod
    def load_dataset(self,
                     root: Optional[str] = None,
                     train: Optional[bool] = True,
                     download: Optional[bool] = True,
                     **kwargs) -> DatasetWrapper:
        """
        Loads the dataset.

        Raises:
            NotImplementedError: If the method is not implemented in the child class.
        """
        pass
