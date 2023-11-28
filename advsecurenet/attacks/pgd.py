import torch

from advsecurenet.attacks.adversarial_attack import AdversarialAttack
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.attack_configs import PgdAttackConfig

"""
This module contains the implementation of the Projected Gradient Descent attack.
"""


class PGD(AdversarialAttack):
    """
    Projected Gradient Descent targeted / untargeted attack using l-infinity norm.

    Args:
        epsilon (float): The epsilon value to use for the attack. Defaults to 0.3.
        alpha (float): The alpha value to use for the attack. Defaults to 2/255.
        num_iter (int): The number of iterations to use for the attack. Defaults to 40.
        device (torch.device): Device to use for the attack. Defaults to "cpu".

    References:
        [1] Madry, Aleksander, et al. "Towards deep learning models resistant to adversarial attacks." arXiv preprint arXiv:1706.06083 (2017).  
    """

    def __init__(self, config: PgdAttackConfig) -> None:
        self.epsilon: float = config.epsilon
        self.alpha: float = config.alpha
        self.num_iter: int = config.num_iter
        super().__init__(config)

    def attack(self, model: BaseModel, x: torch.Tensor, y: torch.Tensor, targeted: bool = False, *args, **kwargs) -> torch.Tensor:
        """
        Performs the PGD attack on the specified model and input.

        Args:
            model (BaseModel): The model to attack.
            x (torch.tensor): The original input tensor. Expected shape is (batch_size, channels, height, width).
            y (torch.tensor): The true labels for the input tensor. Expected shape is (batch_size,).
            targeted (bool): If True, targets the attack to the specified label. Defaults to False.

        Returns:
            torch.tensor: The adversarial example tensor.
        """

        # Random initialization
        delta = torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        delta = torch.clamp(delta, min=-self.epsilon, max=self.epsilon)

        # for _ in trange(self.num_iter, desc=f"{red}PGD Iterations{reset}", bar_format="{l_bar}%s{bar}%s{r_bar}" % (yellow, reset), leave=True):
        for _ in range(self.num_iter):
            delta = self._pgd_step(model, x, y, targeted, delta)

        adv_x = torch.clamp(x + delta, 0, 1)
        return adv_x.detach()

    def _pgd_step(self, model: BaseModel, x: torch.Tensor, y: torch.Tensor, targeted: bool, delta: torch.Tensor) -> torch.Tensor:
        """
        Perform a single PGD step.
        """
        # Make a copy of delta and set its requires_grad attribute
        delta_prime = delta.detach().clone()
        delta_prime.requires_grad = True

        outputs = model(x + delta_prime)
        if targeted:
            # minimize the loss for the target label
            loss = -torch.nn.functional.cross_entropy(outputs, y)
        else:
            # maximize the loss for the correct label
            loss = torch.nn.functional.cross_entropy(outputs, y)

        loss.backward()

        # PGD step
        delta_prime = (delta_prime + self.alpha *
                       delta_prime.grad.detach().sign()).clamp(-self.epsilon, self.epsilon)

        # Projection step
        # keep pixel values in [0,1]
        delta_prime = torch.min(torch.max(delta_prime, -x), 1-x)

        return delta_prime
