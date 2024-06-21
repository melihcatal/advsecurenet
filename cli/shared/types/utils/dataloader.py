from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataLoaderConfig:
    """
    This dataclass is used to store the configuration for a data loader. It can be used generically
    for any type of data loading scenario (e.g., training, testing).
    """
    num_workers: int = 1
    shuffle: bool = False
    drop_last: bool = False
    pin_memory: bool = True
    batch_size: int = 32


@dataclass
class DataLoaderCliConfigType:
    """
    This dataclass encapsulates the data loader configurations for different scenarios.
    It allows for a single default configuration and optional specific configurations.
    """
    default: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    train: Optional[DataLoaderConfig] = None
    test: Optional[DataLoaderConfig] = None
