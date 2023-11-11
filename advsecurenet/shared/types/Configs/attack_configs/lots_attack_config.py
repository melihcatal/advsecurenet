from typing import Union
import torch
from enum import Enum
from dataclasses import dataclass
from advsecurenet.shared.types.configs.attack_configs import AttackConfig


class LotsAttackMode(Enum):
    ITERATIVE = "iterative"
    SINGLE = "single"


@dataclass(kw_only=True)
class LotsAttackConfig(AttackConfig):
    deep_feature_layer: str
    mode: LotsAttackMode = LotsAttackMode.ITERATIVE
    epsilon: float = 0.1
    learning_rate: float = 1./255.
    max_iterations: int = 1000
    verbose: bool = True
    device: torch.device
