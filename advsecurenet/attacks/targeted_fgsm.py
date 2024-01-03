from typing import List

import torch

from advsecurenet.attacks.adversarial_attack import AdversarialAttack
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.attack_configs import FgsmAttackConfig


class TargetedFGSM(AdversarialAttack):
    """
    Targeted Fast Gradient Sign Method attack

    Args:
        epsilon (float): The epsilon value to use for the attack. Defaults to 0.3.
        target_classes (List[int]): The list of target classes for each input in the batch.
        device (torch.device): Device to use for the attack. Defaults to "cpu".

    References:
            [1] Goodfellow, Ian J., et al. "Explaining and harnessing adversarial examples." arXiv preprint arXiv:1412.6572 (2014).
    """

    def __init__(self, config: FgsmAttackConfig) -> None:
        self.epsilon: float = config.epsilon
        super().__init__(config)

    def attack(self,
               model: BaseModel,
               x: torch.Tensor,
               y: torch.Tensor,
               *args, **kwargs) -> torch.Tensor:
        """
        Generates adversarial examples using the targeted FGSM attack.

        Args:
            model (BaseModel): The model to attack.
            x (torch.tensor): The original input tensor. Expected shape is (batch_size, channels, height, width).
            y (torch.tensor): The target labels tensor. Expected shape is (batch_size,).

        Returns:
            torch.tensor: The adversarial example tensor.
        """
        # Create a tensor of the target class labels
        # y = torch.tensor(
        #     self.target_classes, dtype=torch.long, device=x.device)

        # Check if the batch size of x matches the length of target_classes
        if x.shape[0] != len(y):
            raise ValueError(
                "The batch size of x must match the length of target_classes")

        # Get the gradient of the model with respect to the inputs
        x = x.clone().detach()
        x = self.device_manager.to_device(x)
        x.requires_grad = True
        outputs = model(x)
        loss = torch.nn.functional.cross_entropy(outputs, y)
        model.zero_grad()
        loss.backward()

        # Use the gradient to create an adversarial example
        perturbed_image = self._targeted_fgsm_attack(x, x.grad.data)
        return perturbed_image.detach()

    def _targeted_fgsm_attack(self, image: torch.Tensor, data_grad: torch.Tensor) -> torch.Tensor:
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image in the opposite direction of the gradient
        perturbed_image = image - self.epsilon * sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image
