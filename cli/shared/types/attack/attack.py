from dataclasses import dataclass
from typing import Optional

from advsecurenet.shared.types.configs.attack_configs.attack_config import \
    AttackConfig
from cli.shared.types.utils.dataloader import DataLoaderCliConfigType
from cli.shared.types.utils.dataset import AttacksDatasetCliConfigType
from cli.shared.types.utils.device import DeviceCliConfigType
from cli.shared.types.utils.model import ModelCliConfigType


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
class BaseAttackCLIConfigType:
    """
    This dataclass is used to store the configuration of the shared attack CLI configurations.
    """
    model: ModelCliConfigType
    dataset: AttacksDatasetCliConfigType
    dataloader: DataLoaderCliConfigType
    device: DeviceCliConfigType
    attack_procedure: AttackProcedureCliConfigType
    attack_config: AttackConfig
