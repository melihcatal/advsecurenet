from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from advsecurenet.shared.types.configs.attack_configs.attack_config import \
    AttackConfig
from advsecurenet.shared.types.configs.device_config import DeviceConfig
from cli.shared.types.utils.dataloader import DataLoaderCliConfigType
from cli.shared.types.utils.dataset import AttacksDatasetCliConfigType
from cli.shared.types.utils.model import ModelCliConfigType
from cli.shared.types.utils.target import TargetCLIConfigType

# Define a generic type variable for the attack configuration.
T = TypeVar('T', bound=AttackConfig)


@dataclass
class AttackWithNameConfigDict:
    """
    This dataclass is used to store the configuration of an attack with its name.
    """
    name: str
    config: str


@dataclass
class AttackProcedureCliConfigType:
    """
    This dataclass is used to store the configuration of the attack procedure CLI.
    """
    verbose: Optional[bool] = True
    save_result_images: Optional[bool] = False
    result_images_dir: Optional[str] = None
    result_images_prefix: Optional[str] = None


@dataclass
class AttackCLIConfigType(Generic[T]):
    """
    This dataclass is used to store the configuration of the attack CLI.
    """
    attack_parameters: T


@dataclass
class TargetedAttackCLIConfigType(AttackCLIConfigType[T]):
    """
    This dataclass is used to store the configuration of the targeted attack CLI.
    """
    target_parameters: TargetCLIConfigType


@dataclass
class BaseAttackCLIConfigType:
    """
    This dataclass is used to store the configuration of the shared attack CLI configurations.
    """
    model: ModelCliConfigType
    dataset: AttacksDatasetCliConfigType
    dataloader: DataLoaderCliConfigType
    device: DeviceConfig
    attack_procedure: AttackProcedureCliConfigType
    attack_config: AttackCLIConfigType
