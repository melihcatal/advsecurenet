from dataclasses import dataclass
from typing import Optional


@dataclass
class DeviceConfig:
    """
    This dataclass is used to store the configuration of the device.
    """
    use_ddp: Optional[bool] = False
    processor: Optional[str] = "cpu"
    gpu_ids: Optional[str] = None
