from typing import Union
import torch
from dataclasses import dataclass
from advsecurenet.shared.types.configs.attack_configs import AttackConfig


@dataclass(kw_only=True)
class DeepFoolAttackConfig(AttackConfig):
    num_classes: int = 10
    overshoot: float = 0.02
    max_iterations: int = 50
    device: Union[str, torch.device]
