from dataclasses import dataclass

from advsecurenet.shared.types.configs.attack_configs import FgsmAttackConfig
from cli.shared.types.attack import (BaseAttackCLIConfigType,
                                     TargetedAttackCLIConfigType)


@dataclass
class FgsmAttackCLIConfigType(BaseAttackCLIConfigType):
    """
    This dataclass is used to store the configuration of the FGSM attack CLI.
    """
    attack_config: TargetedAttackCLIConfigType[FgsmAttackConfig]
