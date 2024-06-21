from typing import Optional

import torch

from advsecurenet.evaluation.base_evaluator import BaseEvaluator
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.adversarial_evaluators import adversarial_evaluators


class AdversarialEvaluator(BaseEvaluator):
    """
    Composite evaluator that can be used to evaluate multiple metrics at once.

    Args:
        evaluators (Optional[list[str]], optional): List of evaluators to use. If None, all evaluators will be used. Defaults to None. .
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
        self.evaluators = adversarial_evaluators

        # update target models for transferability evaluator
        if "transferability" in self.evaluators:
            self.evaluators["transferability"].target_models = kwargs.get(
                "target_models", [])

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
               target_labels: Optional[torch.Tensor] = None) -> None:
        """
        Updates the evaluator with new data for streaming mode. 
        Args:
            model (BaseModel): The model to evaluate.
            original_images (torch.Tensor): The original images.
            true_labels (torch.Tensor): The true labels of the original images.
            adversarial_images (torch.Tensor): The adversarial images.
            is_targeted (bool, optional): Whether the attack is targeted or not. Defaults to False.
            target_labels (Optional[torch.Tensor], optional): The target labels for the targeted attack. Defaults to None.

        """

        # Dictionary to store the arguments for each evaluator
        evaluators_to_update = {
            "similarity": [original_images, adversarial_images],
            "robustness_gap": [model, original_images, true_labels, adversarial_images],
            "attack_success_rate": [model, original_images, true_labels, adversarial_images, is_targeted, target_labels],
            "perturbation_distance": [original_images, adversarial_images],
            "transferability": [model, original_images, true_labels, adversarial_images]
        }

        for evaluator, args in evaluators_to_update.items():
            if evaluator in self.selected_evaluators:
                self.evaluators[evaluator].update(*args)

        if "perturbation_effectiveness" in self.selected_evaluators:
            asr = self.evaluators["attack_success_rate"].get_results()
            distance_metric = self.kwargs.get("distance_metric", "L0")
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
