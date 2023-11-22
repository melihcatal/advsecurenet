import torch
import cv2
from advsecurenet.models.base_model import BaseModel
from advsecurenet.attacks.adversarial_attack import AdversarialAttack
from advsecurenet.shared.types.configs.attack_configs import DeepFoolAttackConfig


class DeepFool(AdversarialAttack):
    """
    DeepFool attack

    Args:

        num_classes (int): Number of classes in the dataset. Defaults to 10.
        overshoot (float): Overshoot parameter. Defaults to 0.02.
        max_iterations (int): Maximum number of iterations. Defaults to 50.
        device (torch.device): Device to use for the attack. Defaults to "cpu".


    References:
            [1] Moosavi-Dezfooli, Seyed-Mohsen, et al. "Deepfool: a simple and accurate method to fool deep neural networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.  
    """

    def __init__(self, config: DeepFoolAttackConfig) -> None:
        self.num_classes: int = config.num_classes
        self.overshoot: float = config.overshoot
        self.max_iterations: int = config.max_iterations
        super().__init__(config)

    def attack(self,
               model: BaseModel,
               x: torch.tensor,  # (batch_size, channels, height, width)
               y: torch.tensor,
               *args,
               **kwargs
               ) -> torch.tensor:
        """
        Generates adversarial examples using the DeepFool attack.

        Args:
            model (BaseModel): The model to attack.
            x (torch.tensor): The original input tensor. Expected shape is (batch_size, channels, height, width).
            y (torch.tensor): The true labels for the input tensor. Expected shape is (batch_size,).

        Returns:
            torch.tensor: The adversarial example tensor.
        """

        model.eval()
        x = x.detach()
        y = y.detach()
        x = self.device_manager.to_device(x)
        y = self.device_manager.to_device(y)

        x_adv = x.clone()
        for iteration in range(self.max_iterations):
            x_adv.requires_grad = True
            outputs = model(x_adv)

            # Compute logits for the true classes
            correct_logits = outputs.gather(1, y.unsqueeze(1))

            # Subtract the true class logits from all logits
            differences = outputs - correct_logits[:, None]

            # Create a mask to invalidate the differences for the true class
            true_class_mask = torch.arange(
                self.num_classes, device=self.device_manager.get_current_device()) != y[:, None]
            differences = differences * true_class_mask.float()

            # Set the differences for the true class to a high value
            differences[differences == 0] = float('inf')

            # Find the smallest positive difference
            min_differences, _ = differences.min(dim=1)
            min_differences.backward(torch.ones_like(min_differences))

            # Gradient
            gradient = x_adv.grad.data

            # # Gradient Smoothing
            for i in range(gradient.shape[0]):
                for c in range(gradient.shape[1]):  # iterate over channels
                    gradient_np = gradient[i][c].cpu().numpy()
                    gradient_np = cv2.GaussianBlur(gradient_np, (3, 3), 0)
                    gradient[i][c] = self.device_manager.to_device(
                        torch.tensor(gradient_np))

            # Adaptive Overshoot
            current_overshoot = self.overshoot * \
                (1 - (iteration / self.max_iterations))

            # Update adversarial example using GSM and the smoothed gradient
            x_adv = (x_adv + current_overshoot * gradient).detach()

            # Clamping the values to ensure they are within the valid image range
            x_adv = torch.clamp(x_adv, 0, 1)

            # Predict the new images
            outputs = model(x_adv)
            _, new_pred = outputs.max(1)

            # Check if the new images are adversarial
            if (new_pred != y).all():
                break

        return x_adv
