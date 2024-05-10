
from dataclasses import dataclass
from typing import Dict, List, TypedDict

from cli.types.training import TrainingCliConfigType


@dataclass
class AttackConfigDict(TypedDict):
    """
    This class is used as a type hint for the configuration of an adversarial attack.
    """
    config: Dict[str, str]


@dataclass
class AttackWithConfigDict(TypedDict):
    """
    This class is used as a type hint for an adversarial attack with its configuration.
    """
    attack: str
    config: AttackConfigDict


@dataclass
class ModelWithConfigDict(TypedDict):
    """
    This class is used as a type hint for a model with its configuration.
    """
    config: str


@dataclass
class AdversarialTrainingConfig:
    """
    This class is used to store the configuration of the adversarial training.
    """
    models: List[ModelWithConfigDict]
    attacks: List[AttackWithConfigDict]


@dataclass
class ATCliConfigType:
    """
    This class is used as a type hint for the Adversarial Training CLI configuration.
    """
    training: TrainingCliConfigType
    adversarial_training: AdversarialTrainingConfig
