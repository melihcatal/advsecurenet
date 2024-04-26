from dataclasses import dataclass


@dataclass
class DatasetCliConfigType:
    """
    This dataclass is used to store the configuration of the dataset CLI.
    """
    dataset_name: str
    train_dataset_path: str
    test_dataset_path: str
