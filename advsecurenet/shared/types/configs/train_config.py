from dataclasses import dataclass
from typing import Optional, Union

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from advsecurenet.shared.types.configs.device_config import DeviceConfig


@dataclass
class ModelConfig:
    """
    Configuration class for the model.
    """

    model: nn.Module = None


@dataclass
class TrainingProcessConfig:
    """
    Configuration class for the training process.
    """
    train_loader: DataLoader = None
    criterion: Union[str, nn.Module] = "cross_entropy"
    epochs: int = 10
    learning_rate: float = 0.001
    verbose: bool = False


@dataclass
class OptimizationConfig:
    """
    Configuration class for the optimization process.
    """
    optimizer: Union[str, Optimizer] = "adam"
    optimizer_kwargs: Optional[dict] = None
    scheduler: Optional[Union[str, nn.Module]] = None
    scheduler_kwargs: Optional[dict] = None


@dataclass
class CheckpointConfig:
    """ 
    Configuration class for the checkpoint.
    """
    save_checkpoint: bool = False
    save_checkpoint_path: Optional[str] = None
    save_checkpoint_name: Optional[str] = None
    checkpoint_interval: int = 1
    load_checkpoint: bool = False
    load_checkpoint_path: Optional[str] = None


@dataclass
class FinalModelConfig:
    """
    Configuration class for the final model.
    """
    save_final_model: bool = False
    save_model_path: Optional[str] = None
    save_model_name: Optional[str] = None


@dataclass
class TrainConfig(ModelConfig, TrainingProcessConfig, OptimizationConfig, CheckpointConfig, FinalModelConfig, DeviceConfig):
    """
    Dataclass to store the overall training configuration by aggregating other configurations.
    """
