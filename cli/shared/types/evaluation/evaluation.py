from dataclasses import dataclass
from typing import List

from cli.shared.types.utils.dataloader import DataLoaderCliConfigType
from cli.shared.types.utils.dataset import DatasetCliConfigType
from cli.shared.types.utils.device import DeviceCliConfigType
from cli.shared.types.utils.model import ModelCliConfigType


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
