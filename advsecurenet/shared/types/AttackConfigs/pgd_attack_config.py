from dataclasses import dataclass
from advsecurenet.shared.types.device import DeviceType
from advsecurenet.shared.types.AttackConfigs import AttackConfig

@dataclass
class PgdAttackConfig(AttackConfig):
    epsilon: float = 0.3
    alpha: float = 2/255
    num_iter: int = 40
    device : DeviceType = DeviceType.CPU
