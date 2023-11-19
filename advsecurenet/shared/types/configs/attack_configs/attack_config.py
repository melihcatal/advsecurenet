from dataclasses import dataclass, field
import torch
from typing import Any, Union


@dataclass(kw_only=True)
class AttackConfig:
    distributed_mode: bool = False
    device: torch.device = field(default="cpu", repr=False)

    def __setattr__(self, prop, value):
        if prop == "device":
            value = self._check_device(value)
        super().__setattr__(prop, value)

    @staticmethod
    def _check_device(device: Union[str, torch.device]):
        if isinstance(device, str):
            try:
                device = torch.device(device)
            except:
                device = torch.device("cpu")
        return device
