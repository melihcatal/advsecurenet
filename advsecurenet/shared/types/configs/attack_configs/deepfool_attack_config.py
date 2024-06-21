from dataclasses import dataclass

from advsecurenet.shared.types.configs.attack_configs.attack_config import \
    AttackConfig


@dataclass(kw_only=True)
class DeepFoolAttackConfig(AttackConfig):
    """
    DeepFool attack configuration.
    """
    num_classes: int = 10
    overshoot: float = 0.02
    max_iterations: int = 50
