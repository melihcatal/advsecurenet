from dataclasses import dataclass
from typing import List, Optional

from advsecurenet.shared.types.configs.attack_configs.cw_attack_config import \
    CWAttackConfig
from cli.shared.types.attack import (BaseAttackCLIConfigType,
                                     TargetedAttackCLIConfigType)
from cli.shared.types.utils.target import TargetCLIConfigType


@dataclass
class CwAttackCLIConfigType(BaseAttackCLIConfigType):
    """
    This dataclass is used to store the configuration of the CW attack CLI.
    """
    attack_config: TargetedAttackCLIConfigType[CWAttackConfig]
