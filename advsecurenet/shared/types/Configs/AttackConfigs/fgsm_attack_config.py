from dataclasses import dataclass
from advsecurenet.shared.types.device import DeviceType
from advsecurenet.shared.types.Configs.AttackConfigs import AttackConfig

@dataclass
class FgsmAttackConfig(AttackConfig):
    epsilon: float = 0.3
    device: DeviceType = DeviceType.CPU
