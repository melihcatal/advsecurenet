from dataclasses import dataclass
from typing import List

from advsecurenet.shared.types.configs.attack_configs.lots_attack_config import \
    LotsAttackConfig
from cli.types.attacks.attack_base import BaseAttackCLIConfigType


@dataclass
class LOTSAttackCLIConfig(LotsAttackConfig):
    """
    This dataclass is used to store the configuration of the LOTS attack CLI.
    """
    auto_generate_target_images: bool = True
    target_images_dir: str = None
    target_labels: List[int] = None
    maximum_generation_attempts: int = 1000


@dataclass
class LotsAttackCLIConfigType(BaseAttackCLIConfigType):
    """ 
    This class is used as a type hint for the LOTS attack CLI configuration.
    """

    attack_config: LOTSAttackCLIConfig
