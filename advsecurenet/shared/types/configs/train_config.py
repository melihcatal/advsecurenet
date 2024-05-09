from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from advsecurenet.shared.types.configs.device_config import DeviceConfig


@dataclass
class TrainConfig:
    """
    Dataclass to store the training configuration.
    """
    model: nn.Module
    train_loader: DataLoader
    criterion: Union[str, nn.Module] = "cross_entropy"
    optimizer: Union[str, Optimizer] = "adam"
    optimizer_kwargs: Optional[dict] = None
    scheduler: Optional[Union[str, nn.Module]] = None
    scheduler_kwargs: Optional[dict] = None
    epochs: int = 10
    learning_rate: float = 0.001
    save_checkpoint: bool = False
    save_checkpoint_path: Optional[str] = None
    save_checkpoint_name: Optional[str] = None
    checkpoint_interval: int = 1
    load_checkpoint: bool = False
    load_checkpoint_path: Optional[str] = None
    save_final_model: bool = False
    save_model_path: Optional[str] = None
    save_model_name: Optional[str] = None

    use_ddp: bool = False
    gpu_ids: Optional[List[int]] = None
    pin_memory: bool = False
    verbose: bool = False

    processor: torch.device = torch.device("cpu")
