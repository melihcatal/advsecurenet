from dataclasses import dataclass
from typing import List, Optional

from advsecurenet.shared.types.configs.attack_configs.cw_attack_config import \
    CWAttackConfig
from cli.types.attacks.attack_base import BaseAttackCLIConfigType


@dataclass
class CwAttackCLIConfig:
    """
    This dataclass is used to store the configuration of the CW attack.
    """
    target_labels: Optional[List[int]]
    c_init: Optional[float]
    kappa: Optional[float]
    learning_rate: Optional[float]
    max_iterations: Optional[int]
    abort_early: Optional[bool]
    binary_search_steps: Optional[int]
    clip_min: Optional[float]
    clip_max: Optional[float]
    c_lower: Optional[float]
    c_upper: Optional[float]
    patience: Optional[int]


@dataclass
class CwAttackCLIConfigType(BaseAttackCLIConfigType):
    """
    This dataclass is used to store the configuration of the CW attack CLI.
    """
    attack_config: CWAttackConfig
