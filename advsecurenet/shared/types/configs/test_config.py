from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from advsecurenet.models.base_model import BaseModel


@dataclass
class TestConfig:
    """
    This dataclass is used to store the configuration of the test CLI.
    """
    model: BaseModel
    test_loader: DataLoader
    criterion: Union[str, nn.Module] = "cross_entropy"
    device: Optional[torch.device] = torch.device("cpu")

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
