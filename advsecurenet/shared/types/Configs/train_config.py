from dataclasses import dataclass
from advsecurenet.shared.types.device import DeviceType

@dataclass
class TrainConfig:
    model_name: str = "resnet18"
    dataset_name: str = "cifar10"
    epochs: int = 10
    batch_size: int = 32
    # TODO: change lr to learning_rate
    lr: float = 0.1
    optimizer: str = "adam"
    loss: str = "cross_entropy"
    device: DeviceType = DeviceType.CPU
    save_path: str = "saved_models"
    save_name: str = "model"
    