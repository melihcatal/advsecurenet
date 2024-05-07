from dataclasses import dataclass

from advsecurenet.shared.types.configs.attack_configs import AttackConfig


@dataclass(kw_only=True)
class FgsmAttackConfig(AttackConfig):
    """
    FGSM attack configuration.
    """
    epsilon: float = 0.3
