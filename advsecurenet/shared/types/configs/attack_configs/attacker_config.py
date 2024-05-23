from dataclasses import dataclass
from typing import Optional

import torch

from advsecurenet.attacks.adversarial_attack import AdversarialAttack
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
    dataloader: DataLoaderConfig
    device: DeviceConfig
    return_adversarial_images: Optional[bool] = False
