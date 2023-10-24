from dataclasses import dataclass
from advsecurenet.shared.types.device import DeviceType

@dataclass
class FgsmAttackConfig:
    epsilon: float = 0.3
    device: DeviceType = DeviceType.CPU
