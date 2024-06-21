from dataclasses import dataclass
from typing import Optional

from advsecurenet.shared.types.configs.device_config import DeviceConfig
from cli.shared.types.utils.dataloader import DataLoaderCliConfigType
from cli.shared.types.utils.dataset import DatasetCliConfigType
from cli.shared.types.utils.model import ModelCliConfigType


@dataclass
class Training:
    """ 
    This dataclass is used to store the configuration of the training.
    """
    epochs: int
    learning_rate: float
    optimizer: str
    criterion: str
    save_final_model: bool
    save_model_path: str
    save_model_name: str
    save_checkpoint: bool
    save_checkpoint_path: str
    save_checkpoint_name: str
    checkpoint_interval: int
    load_checkpoint: bool
    load_checkpoint_path: str
    verbose: bool
    scheduler: Optional[str] = None
    scheduler_kwargs: Optional[dict] = None
    optimizer_kwargs: Optional[dict] = None


@dataclass
class TrainingCliConfigType:
    """
    This dataclass is used to store the configuration of the training CLI.
    """
    model: ModelCliConfigType
    dataset: DatasetCliConfigType
    dataloader: DataLoaderCliConfigType
    training: Training
    device: DeviceConfig
