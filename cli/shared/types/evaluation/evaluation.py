from dataclasses import dataclass
from typing import List, Optional

from advsecurenet.shared.types.configs.device_config import DeviceConfig
from cli.shared.types.attack.attack import (AttackProcedureCliConfigType,
                                            AttackWithNameConfigDict)
from cli.shared.types.defense.adversarial_training import ModelWithConfigDict
from cli.shared.types.utils.dataloader import DataLoaderCliConfigType
from cli.shared.types.utils.dataset import AttacksDatasetCliConfigType
from cli.shared.types.utils.model import ModelCliConfigType


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
