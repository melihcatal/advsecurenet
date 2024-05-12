from dataclasses import dataclass
from typing import List

from cli.types.dataloader import DataLoaderCliConfigType
from cli.types.dataset import DatasetCliConfigType
from cli.types.device import DeviceCliConfigType
from cli.types.model import ModelCliConfigType


@dataclass
class EvaluationCliConfigType:
    """
    This dataclass is used to store the configuration of the evaluation CLI.
    """
    model: ModelCliConfigType
    dataset: DatasetCliConfigType
    dataloader: DataLoaderCliConfigType
    device: DeviceCliConfigType
    evaluations: List[str]
