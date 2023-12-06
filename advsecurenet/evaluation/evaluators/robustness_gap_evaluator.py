import torch

from advsecurenet.evaluation.base_evaluator import BaseEvaluator
from advsecurenet.models.base_model import BaseModel


class RobustnessGapEvaluator(BaseEvaluator):
    """
    Evaluator for the robustness gap.
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
        clean_accuracy_count = self._calculate_correct(
            model, original_images, true_labels)
        adversarial_accuracy_count = self._calculate_correct(
            model, adversarial_images, true_labels)

        self.total_correct_clean += clean_accuracy_count
        self.total_correct_adv += adversarial_accuracy_count
        self.total_samples += original_images.size(0)

    def get_results(self) -> float:
        """
        Calculates the robustness gap for the streaming session.

        Returns:
            float: The mean robustness gap for the adversarial examples in the streaming session.
        """
        if self.total_samples > 0:
            clean_accuracy = self.total_correct_clean / self.total_samples
            adversarial_accuracy = self.total_correct_adv / self.total_samples
            return clean_accuracy - adversarial_accuracy
        else:
            return 0.0

    def _calculate_correct(self, model: BaseModel, images: torch.Tensor, labels: torch.Tensor) -> int:
        """
        Calculates the number of correctly classified images.
        """
        model.eval()
        with torch.no_grad():
            predictions = model(images)
            predicted_labels = torch.argmax(predictions, dim=1)
            correct_predictions = torch.sum(predicted_labels == labels)
        return correct_predictions.item()
