import torch
from advsecurenet.attacks.adversarial_attack import AdversarialAttack

from advsecurenet.utils import get_device
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types import FgsmAttackConfig


class FGSM(AdversarialAttack):
    """
    Fast Gradient Sign Method attack

    Args:
        epsilon (float): The epsilon value to use for the attack. Defaults to 0.3.
        device (DeviceType): Device to use for the attack. Defaults to DeviceType.CPU.


    References:
            [1] Goodfellow, Ian J., et al. "Explaining and harnessing adversarial examples." arXiv preprint arXiv:1412.6572 (2014).

    """

    def __init__(self, config: FgsmAttackConfig) -> None:
        self.device = config.device.value if config.device is not None else get_device()
        self.epsilon = config.epsilon

    """
    Generates adversarial examples using the FGSM attack.

    Args:
        model (BaseModel): The model to attack.
        x (torch.tensor): The original input tensor. Expected shape is (batch_size, channels, height, width).
        y (torch.tensor): The true labels for the input tensor. Expected shape is (batch_size,).

    Returns:
    
        torch.tensor: The adversarial example tensor.
    """

    def attack(self, model: BaseModel, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Get the gradient of the model with respect to the inputs.
        x = x.clone().detach().to(self.device)
        x.requires_grad = True
        outputs = model(x)
        loss = torch.nn.functional.cross_entropy(outputs, y)
        model.zero_grad()
        loss.backward()

        # Use the gradient to create an adversarial example.
        perturbed_image = self._fgsm_attack(x, x.grad.data)
        return perturbed_image.detach()

    def _fgsm_attack(self, image: torch.Tensor, data_grad: torch.Tensor) -> torch.Tensor:
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + self.epsilon * sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image