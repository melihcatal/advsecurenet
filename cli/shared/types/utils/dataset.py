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
class AttacksDatasetCliConfigType:
    """
    This dataclass is used to store the configuration of the dataset CLI used for attacks. Attacks do not require a separate test/train dataset.
    """
    dataset_name: str
    dataset_part: str
    custom_data_dir: Optional[str]
    random_sample_size: Optional[int]
