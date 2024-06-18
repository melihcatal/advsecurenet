from typing import Tuple

import torch

from advsecurenet.evaluation.base_evaluator import BaseEvaluator
from advsecurenet.models.base_model import BaseModel


class RobustnessGapEvaluator(BaseEvaluator):
    """
    Evaluator for the robustness gap. The robustness gap is the difference between the accuracy of the model on clean and adversarial examples.
    Currently, this metric doesn't support targeted attacks and doesn't have an option to filter out the initially misclassified images.
    """

    def __init__(self):
        self.total_correct_clean = 0
        self.total_correct_adv = 0
        self.total_samples = 0

    def reset(self):
        """
        Resets the evaluator for a new streaming session.
        """
        self.total_correct_clean = 0
        self.total_correct_adv = 0
        self.total_samples = 0

    def update(self, model: BaseModel, original_images: torch.Tensor, true_labels: torch.Tensor, adversarial_images: torch.Tensor):
        """
        Updates the evaluator with new data for streaming mode.

        Args:
            original_images (torch.Tensor): The original images.
            true_labels (torch.Tensor): The true labels for the original images.
            adversarial_images (torch.Tensor): The adversarial images.
        """
        clean_accuracy, adversarial_accuracy = self._calculate_acc(
            model, original_images, adversarial_images, true_labels)

        self.total_correct_clean += clean_accuracy
        self.total_correct_adv += adversarial_accuracy
        self.total_samples += original_images.size(0)

    def get_results(self) -> dict[str, float]:
        """
        Calculates the robustness gap for the streaming session.

        Returns:
            dict[str, float]: The robustness gap for the adversarial examples in the streaming session.
        """
        if self.total_samples > 0:
            clean_accuracy = self.total_correct_clean / self.total_samples
            adversarial_accuracy = self.total_correct_adv / self.total_samples
            return {
                "clean_accuracy": clean_accuracy,
                "adversarial_accuracy": adversarial_accuracy,
                "robustness_gap": clean_accuracy - adversarial_accuracy
            }
        else:
            return {
                "clean_accuracy": 0.0,
                "adversarial_accuracy": 0.0,
                "robustness_gap": 0.0
            }

    def _calculate_acc(self, model: BaseModel, clean_images: torch.Tensor, adv_images: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
        """
        Calculates the number of correctly classified images.
        """
        model.eval()
        with torch.no_grad():
            clean_predictions = model(clean_images)
            adv_predictions = model(adv_images)
            clean_predict_labels = torch.argmax(clean_predictions, dim=1)
            adv_predict_labels = torch.argmax(adv_predictions, dim=1)

            clean_correct_predictions = torch.sum(
                clean_predict_labels == labels)
            adv_correct_predictions = torch.sum(adv_predict_labels == labels)
        return clean_correct_predictions.item(), adv_correct_predictions.item()
