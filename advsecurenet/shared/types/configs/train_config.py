import torch
from torch import nn
from typing import List, Union, Optional
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    model: nn.Module
    train_loader: DataLoader
    criterion: Union[str, nn.Module] = "cross_entropy"
    optimizer: Union[str, Optimizer] = "adam"
    epochs: int = 10
    learning_rate: float = 0.001
    save_checkpoint: bool = False
    save_checkpoint_path: Optional[str] = None
    save_checkpoint_name: Optional[str] = None
    checkpoint_interval: int = 1
    load_checkpoint: bool = False
    load_checkpoint_path: Optional[str] = None
    save_final_model: bool = False
    use_ddp: bool = False
    gpu_ids: Optional[List[int]] = None
    pin_memory: bool = False

    device: torch.device = torch.device("cpu")

    def __setattr__(self, prop, value):
        if prop == "device":
            value = self._check_device(value)
        super().__setattr__(prop, value)

    @staticmethod
    def _check_device(device: Union[str, torch.device]):
        if isinstance(device, str):
            try:
                device = torch.device(device)
            except:
                device = torch.device("cpu")
        return device
