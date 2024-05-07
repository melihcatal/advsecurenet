from dataclasses import dataclass
from enum import Enum

import torch

from advsecurenet.shared.types.configs.attack_configs import AttackConfig


class LotsAttackMode(Enum):
    ITERATIVE = "iterative"
    SINGLE = "single"


@dataclass(kw_only=True)
class LotsAttackConfig(AttackConfig):
    """Configuration class for LotsAttack.

    Attributes:
        deep_feature_layer (str): The deep feature layer to be used.
        mode (LotsAttackMode): The mode of the LotsAttack. Defaults to LotsAttackMode.SINGLE.
        epsilon (float): The epsilon value for the attack. Defaults to 0.1.
        learning_rate (float): The learning rate for the attack. Defaults to 1./255.
        max_iterations (int): The maximum number of iterations for the attack. Defaults to 1000.
        verbose (bool): Whether to print verbose output during the attack. Defaults to True.
        device (torch.device): The device to be used for the attack.
    """
    deep_feature_layer: str
    mode: LotsAttackMode = LotsAttackMode.SINGLE
    epsilon: float = 0.1
    learning_rate: float = 1./255.
    max_iterations: int = 1000
    verbose: bool = True
    device: torch.device
