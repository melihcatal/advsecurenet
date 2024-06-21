from dataclasses import dataclass, field
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader

from advsecurenet.attacks.base.adversarial_attack import AdversarialAttack
from advsecurenet.shared.types.configs.dataloader_config import \
    DataLoaderConfig
from advsecurenet.shared.types.configs.device_config import DeviceConfig


@dataclass
class AttackerConfig:
    """ 
    Configuration class for the Attacker module.
    """
    model: torch.nn.Module
    attack: AdversarialAttack
    dataloader: Union[DataLoader, DataLoaderConfig]
    device: DeviceConfig
    return_adversarial_images: Optional[bool] = False
    evaluators: Optional[list[str]] = field(
        default_factory=lambda: ["attack_success_rate"])
