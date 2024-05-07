from dataclasses import dataclass

from advsecurenet.shared.types.configs.attack_configs.attack_config import \
    AttackConfig
from cli.types.attack import AttackProcedureCliConfigType
from cli.types.dataloader import DataLoaderCliConfigType
from cli.types.dataset import AttacksDatasetCliConfigType
from cli.types.device import DeviceCliConfigType
from cli.types.model import ModelCliConfigType


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
