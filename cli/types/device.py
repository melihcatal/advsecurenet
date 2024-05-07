from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DeviceCliConfigType:
    """
    This dataclass is used to store the configuration of the device.
    """
    use_ddp: bool = field(default=False, init=True)
    device: str = field(default="cpu")
    gpu_ids: Optional[str] = field(default=None)
    pin_memory: Optional[bool] = field(default=False)
