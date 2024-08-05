from dataclasses import dataclass, field

from advsecurenet.shared.types.configs.device_config import DeviceConfig


@dataclass(kw_only=True)
class AttackConfig:
    """
    This dataclass is used to store the base configuration of the attacks. It contains the device configuration and a flag to indicate if the attack is targeted or not.
    """

    device: DeviceConfig = field(default_factory=DeviceConfig)
    targeted: bool = False
