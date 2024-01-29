from typing import Optional

from advsecurenet.evaluation.base_evaluator import BaseEvaluator
from advsecurenet.evaluation.evaluators import (
    AttackSuccessRateEvaluator, PerturbationDistanceEvaluator,
    PerturbationEffectivenessEvaluator, RobustnessGapEvaluator,
    SimilarityEvaluator, TransferabilityEvaluator)
from advsecurenet.utils.data import unnormalize_data


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

        Some metrics expect the adversarial examples to be normalized. Similarity metrics expect the unnormalized data. 
        The metrics that need to feed the model expect the data to be normalized since the model has been trained on normalized data.
        The complete list of metrics and their expected data format is:

            - SimilarityEvaluator: Expects unnormalized data.
            - PerturbationDistanceEvaluator: Expects unnormalized data.
            - PerturbationEffectivenessEvaluator: Expects unnormalized data.
            - RobustnessGapEvaluator: Expects normalized data.
            - AttackSuccessRateEvaluator: Expects normalized data.
            - TransferabilityEvaluator: Expects normalized data.

    """

    def __init__(self,
                 evaluators: Optional[list[str]] = None,
                 mean: Optional[list[float]] = None,
                 std: Optional[list[float]] = None,
                 **kwargs):
        self.mean = mean
        self.std = std
        self.kwargs = kwargs

        # Dictionary to store evaluator instances
        self.evaluators = {
            "similarity": SimilarityEvaluator(),
            "robustness_gap": RobustnessGapEvaluator(),
            "attack_success_rate": AttackSuccessRateEvaluator(),
            "perturbation_effectiveness": PerturbationEffectivenessEvaluator(),
            "perturbation_distance": PerturbationDistanceEvaluator(self.kwargs["normalize"] if "normalize" in self.kwargs else False),
            "transferability": TransferabilityEvaluator(self.kwargs["target_models"] if "target_models" in self.kwargs else [])
        }
        # Filter evaluators based on the provided list
        if evaluators is None:
            self.selected_evaluators = self.evaluators
        else:
            self.selected_evaluators = {
                key: self.evaluators[key] for key in evaluators}
            self._validate_evaluators()

    def reset(self):
        """
        Resets the evaluator for a new streaming session.
        """
        for key in self.selected_evaluators:
            self.evaluators[key].reset()

    def update(self, model, images, labels, adv_img):
        """
        Updates the evaluator with new data for streaming mode. Expects normalized data. If needed, the data will be unnormalized before calculating the metrics.
        """
        if "similarity" in self.selected_evaluators:
            unnormalized_images, unnormalized_adv_images = self._get_unnormalized_data(
                images, adv_img)
            self.evaluators["similarity"].update(
                unnormalized_images, unnormalized_adv_images)
        if "robustness_gap" in self.selected_evaluators:
            self.evaluators["robustness_gap"].update(
                model, images, labels, adv_img)
        if "attack_success_rate" in self.selected_evaluators:
            self.evaluators["attack_success_rate"].update(
                model, images, labels, adv_img)
        if "perturbation_distance" in self.selected_evaluators:
            unnormalized_images, unnormalized_adv_images = self._get_unnormalized_data(
                images, adv_img)
            self.evaluators["perturbation_distance"].update(
                unnormalized_images, unnormalized_adv_images)

        if "transferability" in self.selected_evaluators:
            self.evaluators["transferability"].update(
                model, images, labels, adv_img)

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

    def _get_unnormalized_data(self, images, adv_img):
        images_clone = images.clone()
        adv_img_clone = adv_img.clone()
        unnormalized_images = unnormalize_data(
            images_clone, self.mean, self.std)
        unnormalized_adv_images = unnormalize_data(
            adv_img_clone, self.mean, self.std)
        return unnormalized_images, unnormalized_adv_images

    def _validate_evaluators(self):
        if "similarity" in self.selected_evaluators or "perturbation_distance" in self.selected_evaluators or "perturbation_effectiveness" in self.selected_evaluators:
            assert self.mean is not None and self.std is not None, "Mean and std must be provided for the selected evaluators."
