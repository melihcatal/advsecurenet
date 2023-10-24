from abc import ABC, abstractmethod
from advsecurenet.models.base_model import BaseModel
import torch


class AdversarialAttack(ABC):
    """
    Abstract class for adversarial attacks.
    """

    @abstractmethod
    def attack(self,
               model: BaseModel,
               x: torch.Tensor,
               y: torch.Tensor) -> torch.Tensor:
        """
        Performs the attack on the specified model and input. 

        Args:   
            model (BaseModel): The model to attack.
            x (torch.tensor): The original input tensor. Expected shape is (batch_size, channels, height, width).
            y (torch.tensor): The true labels for the input tensor. Expected shape is (batch_size,).

        Returns:
            torch.tensor: The adversarial example tensor.

        """
        pass
