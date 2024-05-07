from dataclasses import dataclass

from advsecurenet.shared.types.configs.attack_configs.deepfool_attack_config import \
    DeepFoolAttackConfig
from cli.types.attacks.attack_base import BaseAttackCLIConfigType


@dataclass
class DeepFoolAttackCLIConfigType(BaseAttackCLIConfigType):
    """
    This dataclass is used to store the configuration of the DeepFool attack CLI.
    """
    attack_config: DeepFoolAttackConfig
