from dataclasses import dataclass, field
from typing import Union

import torch


@dataclass(kw_only=True)
class TestConfig:
    """
    This dataclass is used to store the configuration of the test CLI.
    """
    model_name: str
    dataset_name: str
    dataset_path: str
    # TODO: change model_weights to model_weight_path
    model_weights: str
    pretrained: bool = False
    batch_size: int = 32
    loss: str = "cross_entropy"
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
