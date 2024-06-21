from dataclasses import dataclass
from typing import Optional, Union

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
    processor: Optional[torch.device] = torch.device("cpu")
    topk: int = 1
