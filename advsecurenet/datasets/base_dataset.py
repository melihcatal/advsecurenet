from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import v2 as transforms

from advsecurenet.shared.types.configs.preprocess_config import (
    PreprocessConfig, PreprocessStep)
from advsecurenet.shared.types.dataset import DataType


class DatasetWrapper(TorchDataset):
    """ 
    A wrapper class for PyTorch datasets that allows for easy access to the underlying dataset and having customized parameters.
    """

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

    Args:
        preprocess_config (Optional[PreprocessConfig], optional): The preprocessing configuration for the dataset. Defaults to None.

    Attributes:
        _dataset (TorchDataset): The underlying PyTorch dataset.
        mean (List[float]): The mean values of the dataset.
        std (List[float]): The standard deviation values of the dataset.
        input_size (Tuple[int, int]): The input size of the dataset.
        name (str): The name of the dataset.
        num_classes (int): The number of classes in the dataset.
        num_input_channels (int): The number of input channels in the dataset.
        data_type (DataType): The type of the dataset (train or test).

    Note:
        This module uses v2 of the torchvision transforms. Please refer to the official PyTorch documentation for more information about the possible transforms.
    """

    def __init__(self,
                 preprocess_config: Optional[PreprocessConfig] = None
                 ):
        self._dataset: DatasetWrapper
        self.mean: List[float] = []
        self.std: List[float] = []
        self.input_size: Tuple[int, int] = ()
        self.crop_size: Tuple[int, int] = ()
        self.name: str = ""
        self.num_classes: int = 0
        self.num_input_channels: int = 0
        self.data_type: DataType = None
        self._preprocess_config = preprocess_config

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

    def get_transforms(self) -> transforms.Compose:
        """
        Returns the data transforms to be applied to the dataset.

        Returns:
            transforms.Compose: The data transforms.
        """
        available_transforms = transforms.__dict__.keys()
        available_transforms = [
            t for t in available_transforms if not t.startswith("_")]

        if self._preprocess_config and self._preprocess_config.steps:
            preprocess_steps = self._preprocess_config.steps
            transform_steps = []
            for preprocess_step in preprocess_steps:
                # convert the preprocess step to a PreprocessStep object
                preprocess_step = PreprocessStep(**preprocess_step)
                # get the name of the transform
                name = preprocess_step.name
                # get the transform function from the v2 module
                transform = getattr(transforms, name)
                # get the parameters provided in the config
                params = preprocess_step.params
                # add the transform to the list of transforms with the provided parameters
                transform_steps.append(transform(**params))

            return transforms.Compose(transform_steps)
        else:
            return transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
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
