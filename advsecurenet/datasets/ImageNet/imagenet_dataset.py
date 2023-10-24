from torchvision import datasets
from torch.utils.data import Dataset as TorchDataset
from advsecurenet.datasets.base_dataset import BaseDataset
from advsecurenet.shared.types import DataType
from typing import Optional


class ImageNetDataset(BaseDataset):
    """
    The ImageNetDataset class that loads the ImageNet dataset.

    Attributes:
        mean (List[float]): Mean of the ImageNet dataset.
        std (List[float]): Standard deviation of the ImageNet dataset.
        input_size (Tuple[int, int]): Input size of the ImageNet images.
        name (str): Name of the dataset.
        num_classes (int): Number of classes in the ImageNet dataset.
        num_input_channels (int): Number of input channels in the ImageNet images.
    """

    def __init__(self):
        super().__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.input_size = (224, 224)
        self.name = "imagenet"
        self.num_classes = 1000
        self.num_input_channels = 3

    def load_dataset(self,
                     root: Optional[str] = './data/imagenet',
                     train: Optional[bool] = True,
                     download: Optional[bool] = False,
                     **kwargs) -> TorchDataset:
        """
        Loads the ImageNet dataset.

        Args:
            root (str, optional): The root directory where the dataset should be stored. Defaults to './data/imagenet'.
            train (bool, optional): If True, assumes the dataset is the training set. Otherwise, assumes it's the validation set. Defaults to True.
            download (bool, optional): If True, would download the dataset from the internet (though this is generally not feasible for ImageNet). Defaults to False.
            **kwargs: Arbitrary keyword arguments for the ImageNet dataset.

        Returns:
            TorchDataset: The ImageNet dataset loaded into memory.
        """
        transform = self.get_transforms()
        self._dataset = datasets.ImageFolder(
            root=root, transform=transform, **kwargs)
        self.data_type = DataType.TRAIN if train else DataType.TEST
        return self._dataset
