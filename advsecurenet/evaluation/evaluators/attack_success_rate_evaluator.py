from typing import Optional

import torch

from advsecurenet.evaluation.base_evaluator import BaseEvaluator


class AttackSuccessRateEvaluator(BaseEvaluator):
    """
    Evaluates the attack success rate for adversarial examples.
    """

    def __init__(self):
        self.total_successful_attacks = 0
        self.total_samples = 0

    def reset(self):
        """
        Resets the evaluator for a new streaming session.
        """
        self.total_successful_attacks = 0
        self.total_samples = 0

    def update(self, model, original_images: torch.Tensor, true_labels: torch.Tensor, adversarial_images: torch.Tensor, is_targeted: bool = False, target_labels: Optional[torch.Tensor] = None):
        """
        Updates the evaluator with new data for streaming mode.

        Args:
            model: The model being evaluated.
            original_images (torch.Tensor): The original images.
            true_labels (torch.Tensor): The true labels for the original images.
            adversarial_images (torch.Tensor): The adversarial images.
            is_targeted (bool, optional): Whether the attack is targeted.
            target_labels (Optional[torch.Tensor], optional): Target labels for the adversarial images if the attack is targeted.
        """
        model.eval()
        # prediction on original images
        clean_predictions = model(original_images)
        clean_prediction_labels = torch.argmax(clean_predictions, dim=1)

        # if the initial prediction is wrong, don't bother attacking, just skip

        predictions = model(adversarial_images)
        labels = torch.argmax(predictions, dim=1)

        if is_targeted:
            if target_labels is None:
                raise ValueError(
                    "Target labels must be provided for targeted attacks.")
            successful = torch.sum((labels == target_labels) & (
                clean_prediction_labels == true_labels))
        else:
            successful = torch.sum((labels != true_labels) & (
                clean_prediction_labels == true_labels))

        self.total_successful_attacks += successful.item()
        self.total_samples += torch.sum(clean_prediction_labels ==
                                        true_labels).item()

    def get_results(self) -> float:
        """
        Calculates the attack success rate for the streaming session.

        Returns:
            float: The attack success rate for the adversarial examples in the streaming session.
        """
        if self.total_samples > 0:
            return self.total_successful_attacks / self.total_samples
        else:
            return 0.0
