from dataclasses import dataclass

from cli.types.dataloader import DataLoaderCliConfigType
from cli.types.dataset import DatasetCliConfigType
from cli.types.device import DeviceCliConfigType
from cli.types.model import ModelCliConfigType


@dataclass
class Testing:
    """
    This dataclass is used to store the configuration of the testing.
    """
    criterion: str


@dataclass
class TestingCliConfigType:
    """
    This dataclass is used to store the configuration of the testing CLI.
    """
    model: ModelCliConfigType
    dataset: DatasetCliConfigType
    dataloader: DataLoaderCliConfigType
    device: DeviceCliConfigType
    testing: Testing
