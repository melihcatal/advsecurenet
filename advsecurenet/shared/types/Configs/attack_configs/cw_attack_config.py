from dataclasses import dataclass
from advsecurenet.shared.types.device import DeviceType
from advsecurenet.shared.types.configs.attack_configs import AttackConfig


@dataclass
class CWAttackConfig(AttackConfig):
    targeted: bool = False
    c_init: float = 0.1
    kappa: float = 0
    learning_rate: float = 0.01
    max_iterations: int = 10
    abort_early: bool = False
    binary_search_steps: int = 10
    device: DeviceType = DeviceType.CPU
    clip_min: float = 0
    clip_max: float = 1
    c_lower: float = 1e-6
    c_upper: float = 1
    patience: int = 5
    verbose: bool = True
