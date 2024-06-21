from dataclasses import dataclass, field

from advsecurenet.shared.types.configs.device_config import DeviceConfig


@dataclass(kw_only=True)
class AttackConfig:
    """
    This dataclass is used to store the base configuration of the attacks.
    """
    device: DeviceConfig = field(default_factory=DeviceConfig)
    targeted: bool = False
