from typing import Optional

from advsecurenet.evaluation.base_evaluator import BaseEvaluator
from advsecurenet.evaluation.evaluators import (
    AttackSuccessRateEvaluator, PerturbationDistanceEvaluator,
    PerturbationEffectivenessEvaluator, RobustnessGapEvaluator,
    SimilarityEvaluator, TransferabilityEvaluator)


class AdversarialEvaluator(BaseEvaluator):
    """
    Composite evaluator that can be used to evaluate multiple metrics at once.

    It's possible to provide a list of target models to evaluate the transferability of the adversarial examples.
    It's also possible to provide a distance metric to evaluate the perturbation effectiveness of the adversarial examples. Possible distance metrics are:
    - L0
    - L2
    - Linf
    Default distance metric is L0.
    """

    def __init__(self, evaluators: Optional[list[str]] = None, **kwargs):
        # Dictionary to store evaluator instances
        self.kwargs = kwargs
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

    def update(self, model, images, labels, adv_img):
        """
        Updates the evaluator with new data for streaming mode.
        """
        if "similarity" in self.selected_evaluators:
            self.evaluators["similarity"].update(images, adv_img)
        if "robustness_gap" in self.selected_evaluators:
            self.evaluators["robustness_gap"].update(
                model, images, labels, adv_img)
        if "attack_success_rate" in self.selected_evaluators:
            self.evaluators["attack_success_rate"].update(
                model, images, labels, adv_img)
        if "perturbation_distance" in self.selected_evaluators:
            self.evaluators["perturbation_distance"].update(images, adv_img)

        if "transferability" in self.selected_evaluators:
            self.evaluators["transferability"].update(
                model, images, labels, adv_img)

        if "perturbation_effectiveness" in self.selected_evaluators:
            asr = self.evaluators["attack_success_rate"].get_results()
            distance_metric = self.kwargs["distance_metric"] if "distance_metric" in self.kwargs else "L0"
            distance_metric_index = self._get_distance_metric_index(
                distance_metric)
            pd = self.evaluators["perturbation_distance"].get_results()[
                distance_metric_index]
            self.evaluators["perturbation_effectiveness"].update(asr, pd)

    def get_results(self) -> dict:
        """
        Calculates the results for the streaming session.
        """
        results = {}
        for key in self.selected_evaluators:
            results[key] = self.evaluators[key].get_results()
        return results

    def _get_distance_metric_index(self, distance_metric: str):
        distance_metrics = {
            "L0": 0,
            "L2": 1,
            "Linf": 2
        }
        return distance_metrics[distance_metric]
