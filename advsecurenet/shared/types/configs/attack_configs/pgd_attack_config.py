from typing import Union
import torch
from dataclasses import dataclass
from advsecurenet.shared.types.configs.attack_configs import AttackConfig


@dataclass(kw_only=True)
class PgdAttackConfig(AttackConfig):
    epsilon: float = 0.3
    alpha: float = 2/255
    num_iter: int = 40
    device: Union[str, torch.device]
