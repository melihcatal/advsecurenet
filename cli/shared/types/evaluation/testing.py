from dataclasses import dataclass

from advsecurenet.shared.types.configs.device_config import DeviceConfig
from cli.shared.types.utils.dataloader import DataLoaderCliConfigType
from cli.shared.types.utils.dataset import DatasetCliConfigType
from cli.shared.types.utils.model import ModelCliConfigType


@dataclass
class Testing:
    """
    This dataclass is used to store the configuration of the testing.
    """
    criterion: str
    topk: int


@dataclass
class TestingCliConfigType:
    """
    This dataclass is used to store the configuration of the testing CLI.
    """
    model: ModelCliConfigType
    dataset: DatasetCliConfigType
    dataloader: DataLoaderCliConfigType
    device: DeviceConfig
    testing: Testing
