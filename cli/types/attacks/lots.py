from dataclasses import dataclass

from advsecurenet.shared.types.configs.attack_configs.lots_attack_config import \
    LotsAttackConfig
from cli.types.attacks.attack_base import BaseAttackCLIConfigType


@dataclass
class LotsAttackCLIConfigType(BaseAttackCLIConfigType):
    """ 
    This class is used as a type hint for the LOTS attack CLI configuration.
    """

    attack_config: LotsAttackConfig
