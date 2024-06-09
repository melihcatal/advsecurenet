from typing import Optional, Tuple

import torch

from advsecurenet.evaluation.base_evaluator import BaseEvaluator
from advsecurenet.models.base_model import BaseModel


class AttackSuccessRateEvaluator(BaseEvaluator):
    """
    Evaluates the attack success rate for adversarial examples.
    Note:
        This evaluation only considers the samples where the model's initial prediction is correct. This is to ensure that the metrics are not skewed by incorrect initial predictions.
        The results are calculated as the number of successful attacks divided by the total number of samples. The range of the results is [0, 1] where 1 indicates that all the attacks were successful and 0 indicates that none of the attacks were successful or there were no samples to evaluate.
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

    def get_results(self) -> float:
        """
        Calculates the attack success rate for the streaming session.

        Returns:
            float: The attack success rate for the adversarial examples in the streaming session.
        """
        if self.total_samples > 0:
            return self.total_successful_attacks / self.total_samples
        return 0.0

    def update(self,
               model: BaseModel,
               original_images: torch.Tensor,
               true_labels: torch.Tensor,
               adversarial_images: torch.Tensor,
               is_targeted: Optional[bool] = False,
               target_labels: Optional[torch.Tensor] = None):
        """
        Updates the evaluator with new data for streaming mode.

        Args:
            model (BaseModel): The model being evaluated.
            original_images (torch.Tensor): The original images.
            true_labels (torch.Tensor): The true labels for the original images.
            adversarial_images (torch.Tensor): The adversarial images.
            is_targeted (bool, optional): Whether the attack is targeted.
            target_labels (Optional[torch.Tensor], optional): Target labels for the adversarial images if the attack is targeted.

        Note:
            This function only considers the samples where the model's initial prediction is correct. This is to ensure that the metrics are not skewed by incorrect initial predictions.
        """
        correct_target_labels = None
        model.eval()

        clean_prediction_labels = self._predict(model, original_images)
        correct_initial_predictions_mask = self._get_correct_predictions_mask(
            clean_prediction_labels, true_labels)
        total_correct_initial = correct_initial_predictions_mask.sum().item()
        # If there are no correct initial predictions, no need to evaluate
        if total_correct_initial == 0:
            return

        correct_true_labels, correct_adversarial_images = self._filter_correct_predictions(
            correct_initial_predictions_mask, true_labels, adversarial_images)

        if is_targeted:
            self._validate_targeted_attack(target_labels)
            correct_target_labels = target_labels[correct_initial_predictions_mask]

        adversarial_labels = self._predict(model, correct_adversarial_images)

        successful_attacks = self._calculate_success(
            adversarial_labels, correct_true_labels, is_targeted, correct_target_labels if is_targeted else None)

        self._update_metrics(successful_attacks, total_correct_initial)

    def _predict(self,
                 model: BaseModel,
                 images: torch.Tensor) -> torch.Tensor:
        """ 
        Predicts the labels for the given images using the model.

        Args:
            model (BaseModel): The model to use for prediction.
            images (torch.Tensor): The images to predict the labels for.

        Returns:
            torch.Tensor: The predicted labels for the images.
        """
        predictions = model(images)
        prediction_labels = torch.argmax(predictions, dim=1)
        return prediction_labels

    def _get_correct_predictions_mask(self, prediction_labels: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
        """
        Returns a mask for the samples where the model's initial prediction is correct. The mask is True for correct predictions and False for incorrect predictions.

        Args:
            prediction_labels (torch.Tensor): The predicted labels for the samples.
            true_labels (torch.Tensor): The true labels for the samples.

        Returns:
            torch.Tensor: A mask for the samples where the model's initial prediction is correct.
        """
        return prediction_labels == true_labels

    def _filter_correct_predictions(self, mask: torch.Tensor, true_labels: torch.Tensor, adversarial_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filters the correct predictions from the given samples using the mask. 

        Args:
            mask (torch.Tensor): A mask for the samples where the model's initial prediction is correct.
            true_labels (torch.Tensor): The true labels for the samples.
            adversarial_images (torch.Tensor): The adversarial images.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The correct true labels and adversarial images.
        """
        correct_true_labels = true_labels[mask]
        correct_adversarial_images = adversarial_images[mask]
        return correct_true_labels, correct_adversarial_images

    def _validate_targeted_attack(self, target_labels: Optional[torch.Tensor]) -> None:
        """ 
        Validates if the target labels are provided for targeted attacks.

        Args:
            target_labels (Optional[torch.Tensor]): The target labels for the adversarial images.

        Raises:
            ValueError: If the target labels are not provided for targeted attacks.
        """
        if target_labels is None:
            raise ValueError(
                "Target labels must be provided for targeted attacks.")

    def _calculate_success(self, adversarial_labels: torch.Tensor, true_labels: torch.Tensor, is_targeted: Optional[bool] = False, target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ 
        Calculates the number of successful attacks. If the attack is targeted, the function calculates the number of samples where the adversarial labels are equal to the target labels. If the attack is untargeted, the function calculates the number of samples where the adversarial labels are not equal to the true labels.

        Args:
            adversarial_labels (torch.Tensor): The predicted labels for the adversarial images.
            true_labels (torch.Tensor): The true labels for the original images.
            is_targeted (Optional[bool], optional): Whether the attack is targeted. The default is False.
            target_labels (Optional[torch.Tensor], optional): The target labels for the adversarial images if the attack is targeted.

        Returns:
            torch.Tensor (int): The number of successful attacks.
        """
        if is_targeted:
            return torch.sum(adversarial_labels == target_labels, dtype=torch.int)
        else:
            return torch.sum(adversarial_labels != true_labels, dtype=torch.int)

    def _update_metrics(self, successful_attacks: torch.Tensor, total_correct_initial: int) -> None:
        """ 
        Update the evaluator metrics with the results of the current batch.

        Args:
            successful_attacks (torch.Tensor): The number of successful attacks in the current batch.
            total_correct_initial (int): The total number of samples with correct initial predictions.
        """
        self.total_successful_attacks += successful_attacks.item()
        self.total_samples += total_correct_initial
