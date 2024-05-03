from dataclasses import dataclass


@dataclass
class DataLoaderCliConfigType:
    """
    This dataclass is used to store the configuration of the data loader CLI.
    """
    num_workers_train: int
    num_workers_test: int
    shuffle_train: bool
    shuffle_test: bool
    drop_last_train: bool
    drop_last_test: bool
    pin_memory: bool
    batch_size: int
