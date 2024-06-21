from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch

from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.attack_configs.attack_config import \
    AttackConfig
from advsecurenet.utils.device_manager import DeviceManager


class AdversarialAttack(ABC):
    """
    Abstract class for adversarial attacks.
    """

    def __init__(self, config: AttackConfig) -> None:
        self.device_manager = DeviceManager(
            device=config.device.processor,
            distributed_mode=config.device.use_ddp)
        self.name: str = self.__class__.__name__
        self.targeted: bool = config.targeted

    @abstractmethod
    def attack(self,
               model: BaseModel,
               x: torch.Tensor,
               y: torch.Tensor,
               *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, bool]]:
        """
        Performs the attack on the specified model and input.

        Args:
            model (BaseModel): The model to attack.
            x (torch.tensor): The original input tensor. Expected shape is (batch_size, channels, height, width).
            y (torch.tensor): The true labels for the input tensor. Expected shape is (batch_size,).

        Returns:
            torch.tensor: The adversarial example tensor.
            Optional[bool]: True if the attack was successful, False otherwise. This is specially used in LOTS attack.

        """
