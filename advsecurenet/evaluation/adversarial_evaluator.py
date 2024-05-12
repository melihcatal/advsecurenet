from typing import Optional

import torch

from advsecurenet.evaluation.base_evaluator import BaseEvaluator
from advsecurenet.evaluation.evaluators import (
    AttackSuccessRateEvaluator, PerturbationDistanceEvaluator,
    PerturbationEffectivenessEvaluator, RobustnessGapEvaluator,
    SimilarityEvaluator, TransferabilityEvaluator)
from advsecurenet.models.base_model import BaseModel


class AdversarialEvaluator(BaseEvaluator):
    """
    Composite evaluator that can be used to evaluate multiple metrics at once.

    Args:
        evaluators (Optional[list[str]], optional): List of evaluators to use. If None, all evaluators will be used. Defaults to None. 
        mean (Optional[list[float]], optional): Mean of the dataset. Defaults to None. Needed for evaluators that need to unnormalize the data.
        std (Optional[list[float]], optional): Standard deviation of the dataset. Defaults to None. Needed for evaluators that need to unnormalize the data.
        **kwargs: Arbitrary keyword arguments for the evaluators.

    Note:
        It's possible to provide a list of target models to evaluate the transferability of the adversarial examples.
        It's also possible to provide a distance metric to evaluate the perturbation effectiveness of the adversarial examples. Possible distance metrics are:
        - L0
        - L2
        - Linf
        Default distance metric is L0.

    """

    def __init__(self,
                 evaluators: Optional[list[str]] = None,
                 **kwargs):
        self.kwargs = kwargs

        # Dictionary to store evaluator instances
        self.evaluators = {
            "similarity": SimilarityEvaluator(),
            "robustness_gap": RobustnessGapEvaluator(),
            "attack_success_rate": AttackSuccessRateEvaluator(),
            "perturbation_effectiveness": PerturbationEffectivenessEvaluator(),
            "perturbation_distance": PerturbationDistanceEvaluator(),
            "transferability": TransferabilityEvaluator(self.kwargs["target_models"] if "target_models" in self.kwargs else [])
        }
        # Filter evaluators based on the provided list
        if evaluators is None:
            self.selected_evaluators = self.evaluators
        else:
            self.selected_evaluators = {
                key: self.evaluators[key] for key in evaluators}

    def reset(self):
        """
        Resets the evaluator for a new streaming session.
        """
        for key in self.selected_evaluators:
            self.evaluators[key].reset()

    def update(self,
               model: BaseModel,
               original_images: torch.Tensor,
               true_labels: torch.Tensor,
               adversarial_images: torch.Tensor,
               is_targeted: bool = False,
               target_labels: Optional[torch.Tensor] = None):
        """
        Updates the evaluator with new data for streaming mode. Expects normalized data. If needed, the data will be unnormalized before calculating the metrics.
        """
        if "similarity" in self.selected_evaluators:
            self.evaluators["similarity"].update(
                original_images, adversarial_images)
        if "robustness_gap" in self.selected_evaluators:
            self.evaluators["robustness_gap"].update(
                model, original_images, true_labels, adversarial_images)
        if "attack_success_rate" in self.selected_evaluators:
            self.evaluators["attack_success_rate"].update(
                model, original_images, true_labels, adversarial_images, is_targeted, target_labels)
        if "perturbation_distance" in self.selected_evaluators:
            self.evaluators["perturbation_distance"].update(
                original_images, adversarial_images)

        if "transferability" in self.selected_evaluators:
            self.evaluators["transferability"].update(
                model, original_images, true_labels, adversarial_images)

        if "perturbation_effectiveness" in self.selected_evaluators:
            asr = self.evaluators["attack_success_rate"].get_results()
            distance_metric = self.kwargs["distance_metric"] if "distance_metric" in self.kwargs else "L0"
            pd = self.evaluators["perturbation_distance"].get_results()[
                distance_metric]
            self.evaluators["perturbation_effectiveness"].update(asr, pd)

    def get_results(self) -> dict:
        """
        Calculates the results for the streaming session.
        """
        results = {}
        for key in self.selected_evaluators:
            results[key] = self.evaluators[key].get_results()
        return results
