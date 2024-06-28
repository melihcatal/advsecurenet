
from dataclasses import dataclass
from typing import Dict, List, TypedDict

from cli.shared.types.train import TrainingCliConfigType


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
    attacks: List[Dict[str, str]]


@dataclass
class ATCliConfigType:
    """
    This class is used as a type hint for the Adversarial Training CLI configuration.
    """
    training: TrainingCliConfigType
    adversarial_training: AdversarialTrainingConfig
