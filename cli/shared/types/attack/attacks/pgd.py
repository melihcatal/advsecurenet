from dataclasses import dataclass

from advsecurenet.shared.types.configs.attack_configs import PgdAttackConfig
from cli.shared.types.attack import BaseAttackCLIConfigType


@dataclass
class PgdAttackCLIConfigType(BaseAttackCLIConfigType):
    """
    This dataclass is used to store the configuration of the PGD attack CLI.
    """
    attack_config: PgdAttackConfig
