from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DeviceConfig:
    """
    This dataclass is used to store the configuration of the device.
    """
    use_ddp: bool = field(default=False, init=True)
    processor: str = field(default="cpu")
    gpu_ids: Optional[List[int]] = field(default=None)
