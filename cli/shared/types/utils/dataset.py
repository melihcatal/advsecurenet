from dataclasses import dataclass
from typing import Optional

from advsecurenet.shared.types.configs.preprocess_config import \
    PreprocessConfig


@dataclass
class DatasetCliConfigType:
    """
    This dataclass is used to store the configuration of the dataset CLI.
    """
    dataset_name: str
    num_classes: int
    train_dataset_path: Optional[str] = None
    test_dataset_path: Optional[str] = None
    download: Optional[bool] = True
    preprocessing: Optional[PreprocessConfig] = None


@dataclass
class AttacksDatasetCliConfigType(DatasetCliConfigType):
    """
    This dataclass is used to store the configuration of the dataset CLI used for attacks. It extends the DatasetCliConfigType. In addition to the attributes of the DatasetCliConfigType, it has the following attributes:

    Attributes:
        dataset_part (Optional[str]): The part of the dataset to be used for the attack. it can be train, test or all. This is valid if the dataset paths are not provided.
        random_sample_size (Optional[int]): The size of the random sample to be taken from the dataset.

    """
    dataset_part: Optional[str] = "test"
    random_sample_size: Optional[int] = None
