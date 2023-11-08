from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
from advsecurenet.shared.types.device import DeviceType


@dataclass
class TrainConfig:
    model: nn.Module = None
    train_loader: DataLoader = None
    criterion: str or nn.Module = "cross_entropy"
    optimizer: str or optim = "adam"
    epochs: int = 10
    learning_rate: float = 0.001
    device: DeviceType = DeviceType.CPU
    save_checkpoint: bool = False
    save_checkpoint_path: str = None
    save_checkpoint_name: str = None
    checkpoint_interval: int = 1
    load_checkpoint: bool = False
    load_checkpoint_path: str = None
