import torch
from typing import Union
from dataclasses import dataclass


@dataclass(kw_only=True)
class DefenseConfig:
    """
    Abstract class for defense configurations.
    """

    device: torch.device = torch.device("cpu")

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
