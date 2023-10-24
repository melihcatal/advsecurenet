from dataclasses import dataclass
from advsecurenet.shared.types.device import DeviceType

@dataclass
class DeepFoolAttackConfig:
    num_classes: int = 10
    overshoot: float = 0.02
    max_iterations: int = 50
    device: DeviceType = DeviceType.CPU