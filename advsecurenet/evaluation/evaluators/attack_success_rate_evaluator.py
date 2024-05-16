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

        Note:
            This e
        """
        model.eval()

        # Prediction on original images
        clean_predictions = model(original_images)
        clean_prediction_labels = torch.argmax(clean_predictions, dim=1)

        # Mask to identify correct initial predictions
        correct_initial_predictions_mask = (
            clean_prediction_labels == true_labels)

        # Calculate total samples based on correct initial predictions
        total_correct_initial = correct_initial_predictions_mask.sum().item()

        # If no correct initial predictions, skip the update
        if total_correct_initial == 0:
            return

        # Filter the data based on the correct initial predictions
        correct_true_labels = true_labels[correct_initial_predictions_mask]
        correct_adversarial_images = adversarial_images[correct_initial_predictions_mask]

        if is_targeted:
            if target_labels is None:
                raise ValueError(
                    "Target labels must be provided for targeted attacks.")
            correct_target_labels = target_labels[correct_initial_predictions_mask]

        # Prediction on adversarial images
        adversarial_predictions = model(correct_adversarial_images)
        adversarial_labels = torch.argmax(adversarial_predictions, dim=1)

        if is_targeted:
            successful_attacks = torch.sum(
                adversarial_labels == correct_target_labels)
        else:
            successful_attacks = torch.sum(
                adversarial_labels != correct_true_labels)

        self.total_successful_attacks += successful_attacks.item()
        self.total_samples += total_correct_initial

    def get_results(self) -> float:
        """
        Calculates the attack success rate for the streaming session.

        Returns:
            float: The attack success rate for the adversarial examples in the streaming session.
        """
        if self.total_samples > 0:
            return self.total_successful_attacks / self.total_samples
        return 0.0
