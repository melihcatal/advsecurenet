from typing import Optional

from advsecurenet.datasets.base_dataset import (BaseDataset,
                                                ImageFolderBaseDataset)
from advsecurenet.shared.types.configs.preprocess_config import \
    PreprocessConfig


class CustomDataset(ImageFolderBaseDataset, BaseDataset):
    """
    Custom dataset class that loads a custom dataset. It is used to load a dataset that is not part of the standard.

    The expected directory structure is ImageFolder format:
    ```
    custom_dataset/
    ├── class1/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── class2/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...

    ```
    """

    def __init__(self, preprocess_config: Optional[PreprocessConfig] = None):
        ImageFolderBaseDataset.__init__(self)
        BaseDataset.__init__(self, preprocess_config)
        self.name = "custom"
