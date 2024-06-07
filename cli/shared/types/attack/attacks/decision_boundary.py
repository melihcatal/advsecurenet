
from dataclasses import dataclass

from advsecurenet.shared.types.configs.attack_configs.decision_boundary_attack_config import \
    DecisionBoundaryAttackConfig
from cli.shared.types.attack import (BaseAttackCLIConfigType,
                                     TargetedAttackCLIConfigType)


@dataclass
class DecisionBoundaryAttackCLIConfigType(BaseAttackCLIConfigType):
    """
    This dataclass is used to store the configuration of the DecisionBoundary attack CLI.
    """
    attack_config: TargetedAttackCLIConfigType[DecisionBoundaryAttackConfig]
