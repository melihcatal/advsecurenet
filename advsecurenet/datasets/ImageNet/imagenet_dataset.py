from typing import Optional

from advsecurenet.datasets.base_dataset import (BaseDataset,
                                                ImageFolderBaseDataset)
from advsecurenet.shared.types.configs.preprocess_config import \
    PreprocessConfig


class ImageNetDataset(ImageFolderBaseDataset, BaseDataset):
    """s
    The ImageNetDataset class that loads the ImageNet dataset.

    Attributes:
        mean (List[float]): Mean of the ImageNet dataset.
        std (List[float]): Standard deviation of the ImageNet dataset.
        input_size (Tuple[int, int]): Input size of the ImageNet images.
        name (str): Name of the dataset.
        num_classes (int): Number of classes in the ImageNet dataset.
        num_input_channels (int): Number of input channels in the ImageNet images.
    """

    def __init__(self, preprocess_config: Optional[PreprocessConfig] = None):
        ImageFolderBaseDataset.__init__(self)
        BaseDataset.__init__(self, preprocess_config)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.input_size = (256, 256)
        self.crop_size = (224, 224)
        self.name = "imagenet"
        self.num_classes = 1000
        self.num_input_channels = 3
