from dataclasses import dataclass
from advsecurenet.shared.types.device import DeviceType

@dataclass
class PgdAttackConfig:
    epsilon: float = 0.3
    alpha: float = 2/255
    num_iter: int = 40
    device : DeviceType = DeviceType.CPU
