from dataclasses import dataclass
from typing import List

from advsecurenet.attacks.base.adversarial_attack import AdversarialAttack
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.train_config import TrainConfig


@dataclass(kw_only=True)
class AdversarialTrainingConfig(TrainConfig):
    """
    This class is used to store the configuration of the adversarial training defense.
    """
    models: List[BaseModel]
    attacks: List[AdversarialAttack]
