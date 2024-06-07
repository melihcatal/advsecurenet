from dataclasses import dataclass
from typing import List, Optional

from advsecurenet.shared.types.configs.device_config import DeviceConfig
from cli.shared.types.defense.adversarial_training import ModelWithConfigDict
from cli.shared.types.utils.dataloader import DataLoaderCliConfigType
from cli.shared.types.utils.dataset import AttacksDatasetCliConfigType
from cli.shared.types.utils.model import ModelCliConfigType


@dataclass
class AttackWithNameConfigDict:
    """
    This dataclass is used to store the configuration of an attack with its name.
    """
    name: str
    config: str


@dataclass
class AdversarialEvaluationConfigType:
    """
    This dataclass is used to store the configuration of the evaluation.
    """
    target_models: List[ModelWithConfigDict]
    attack: AttackWithNameConfigDict
    evaluators: List[str]
    save_results: Optional[bool]
    save_path: Optional[str]
    save_filename: Optional[str]


@dataclass
class AdversarialEvaluationCliConfigType:
    """
    This dataclass is used to store the configuration of the evaluation CLI.
    """
    model: ModelCliConfigType
    dataset: AttacksDatasetCliConfigType
    dataloader: DataLoaderCliConfigType
    device: DeviceConfig
    evaluation: AdversarialEvaluationConfigType
