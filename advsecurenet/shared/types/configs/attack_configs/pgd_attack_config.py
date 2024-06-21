from dataclasses import dataclass

from advsecurenet.shared.types.configs.attack_configs.attack_config import \
    AttackConfig


@dataclass(kw_only=True)
class PgdAttackConfig(AttackConfig):
    """
    PGD attack configuration.
    """
    epsilon: float = 0.3
    alpha: float = 2/255
    num_iter: int = 40
