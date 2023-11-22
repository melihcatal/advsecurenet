from typing import Union
import torch
from dataclasses import dataclass
from advsecurenet.shared.types.configs.attack_configs import AttackConfig


@dataclass(kw_only=True)
class FgsmAttackConfig(AttackConfig):
    epsilon: float = 0.3
