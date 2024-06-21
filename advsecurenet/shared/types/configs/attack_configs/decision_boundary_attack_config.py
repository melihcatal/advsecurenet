from dataclasses import dataclass

from advsecurenet.shared.types.configs.attack_configs.attack_config import \
    AttackConfig


@dataclass(kw_only=True)
class DecisionBoundaryAttackConfig(AttackConfig):
    initial_delta: float = 0.1
    initial_epsilon: float = 0.1
    max_delta_trials: int = 20
    max_epsilon_trials: int = 20
    max_iterations: int = 1000
    max_initialization_trials: int = 100
    step_adapt: float = 0.9
    targeted: bool = False
    verbose: bool = False
    early_stopping: bool = True
    early_stopping_threshold: float = 0.0001
    early_stopping_patience: int = 10
