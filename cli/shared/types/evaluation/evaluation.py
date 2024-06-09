from dataclasses import dataclass
from typing import List

from cli.shared.types.attack.attack import AttackWithNameConfigDict
from cli.shared.types.defense.adversarial_training import ModelWithConfigDict


@dataclass
class AdversarialEvaluationConfigType:
    """
    This dataclass is used to store the configuration of the evaluation.
    """
    target_models: List[ModelWithConfigDict]
    attack: AttackWithNameConfigDict
    evaluators: List[str]


@dataclass
class AdversarialEvaluationCliConfigType:
    """
    This dataclass is used to store the configuration of the evaluation CLI.
    """
    evaluation_config: AdversarialEvaluationConfigType
