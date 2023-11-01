from dataclasses import dataclass
from advsecurenet.shared.types.device import DeviceType

@dataclass
class TestConfig:
    model_name: str = None
    dataset_name: str = None
    # TODO: change model_weights to model_weight_path
    model_weights: str = None
    batch_size: int = 32
    device: DeviceType = DeviceType.CPU
    loss : str = "cross_entropy"