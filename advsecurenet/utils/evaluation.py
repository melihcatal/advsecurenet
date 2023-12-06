import copy
import csv
import os
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from advsecurenet.models.base_model import BaseModel


class AdversarialAttackEvaluator:
    """
    Evaluates the effectiveness of adversarial attacks on a given model using various metrics.

    Args:
        model (BaseModel): The model to evaluate the attacks on.
    """

    def full_evaluation(self,
                        model: BaseModel,
                        original_images: torch.Tensor,
                        true_labels: torch.Tensor,
                        adversarial_images: torch.Tensor,
                        is_targeted: bool = False,
                        target_labels: Optional[torch.Tensor] = None,
                        target_model: Optional[BaseModel] = None,
                        experiment_info: Optional[dict] = None,
                        csv_path: Optional[str] = None,
                        file_name: Optional[str] = None,
                        save_to_csv: bool = False
                        ) -> dict:
        """
        Performs a full evaluation of the adversarial examples using various metrics.

        Args:
            original_images (torch.Tensor): The original images.
            true_labels (torch.Tensor): The true labels for the original images.
            adversarial_images (torch.Tensor): The adversarial images.
            is_targeted (bool, optional): Whether the attack is targeted. Defaults to False.
            target_labels (Optional[torch.Tensor], optional): The target labels for the adversarial images if the attack is targeted. Defaults to None.
            source_model (Optional[BaseModel], optional): The model used to generate adversarial examples. Defaults to None.
            target_model (Optional[BaseModel], optional): A different model to test the transferability of adversarial examples. Defaults to None.
            experiment_info (Optional[dict], optional): The experiment info. Defaults to None.
            csv_path (Optional[str], optional): The path where the CSV file will be saved. Defaults to None.
            file_name (Optional[str], optional): The name of the CSV file to save the results to. Defaults to None. 

        Raises:
            ValueError: If the attack is targeted but target labels are not provided.

        Returns:
            dict: The evaluation results.
        """
        if is_targeted and target_labels is None:
            raise ValueError(
                "Target labels must be provided for targeted attacks.")
        if target_model is None:
            target_model = copy.deepcopy(model)
        if experiment_info is None:
            experiment_info = {}
        if file_name is None:
            file_name = "evaluation_results.csv"

        # Calculate the attack success rate
        attack_success_rate = self.evaluate_attack(model,
                                                   original_images, true_labels, adversarial_images, is_targeted, target_labels)

        # Calculate the L0, L2, and L∞ distances
        l0_distance, l2_distance, l_inf_distance = self.calculate_perturbation_distances(
            original_images, adversarial_images)

        # Calculate the perturbation effectiveness score
        l0_perturbation_effectiveness_score = self.calculate_perturbation_effectiveness_score(
            attack_success_rate, l0_distance)
        l2_perturbation_effectiveness_score = self.calculate_perturbation_effectiveness_score(
            attack_success_rate, l2_distance)
        l_inf_perturbation_effectiveness_score = self.calculate_perturbation_effectiveness_score(
            attack_success_rate, l_inf_distance)

        # Calculate the robustness gap
        robustness_gap = self.calculate_robustness_gap(model,
                                                       original_images, true_labels, adversarial_images)

        # Calculate the SSIM and PSNR
        ssim_score, psnr_score = self.calculate_similarity_scores(
            original_images, adversarial_images)

        # Calculate the transferability rate
        transferability_rate = self.calculate_transferability_rate(
            model, target_model, original_images, true_labels, adversarial_images, is_targeted, target_labels)

        # Create a dictionary of the results
        evaluation_results = {
            "Attack success rate": attack_success_rate,
            "L0 distance": l0_distance,
            "L2 distance": l2_distance,
            "Linf distance": l_inf_distance,
            "L0 perturbation effectiveness score": l0_perturbation_effectiveness_score,
            "L2 perturbation effectiveness score": l2_perturbation_effectiveness_score,
            "Li perturbation effectiveness score": l_inf_perturbation_effectiveness_score,
            "Robustness gap": robustness_gap,
            "SSIM": ssim_score,
            "PSNR": psnr_score,
            "Transferability rate": transferability_rate
        }

        # Save the results to a CSV file
        if save_to_csv:
            self.save_results_to_csv(
                evaluation_results, experiment_info, csv_path, file_name)

        return evaluation_results

    def evaluate_attack(self,
                        model: BaseModel,
                        original_images: torch.Tensor,
                        true_labels: torch.Tensor,
                        adversarial_images: torch.Tensor,
                        is_targeted: bool = False,
                        target_labels: Optional[torch.Tensor] = None,
                        ) -> float:
        """
        Evaluates the attack success rate of the adversarial examples. The attack success rate is the percentage of adversarial examples that are misclassified by the model. 
        If the attack is targeted, the attack success rate is the percentage of adversarial examples that are classified as the target class.

        Args:
            original_images (torch.Tensor): The original images.
            true_labels (torch.Tensor): The true labels for the original images.
            adversarial_images (torch.Tensor): The adversarial images.
            is_targeted (bool, optional): Whether the attack is targeted. Defaults to False.
            target_labels (Optional[torch.Tensor], optional): The target labels for the adversarial images. Required if the attack is targeted. Defaults to None.

        Raises:
            ValueError: If the attack is targeted but target labels are not provided.
        """
        if is_targeted and target_labels is None:
            raise ValueError(
                "Target labels must be provided for targeted attacks.")
        model.eval()  # Set the model to evaluation mode
        adv_predictions = model(adversarial_images)
        adv_labels = torch.argmax(adv_predictions, dim=1)
        if is_targeted:
            successful_attacks = torch.sum(adv_labels == target_labels)
        else:
            successful_attacks = torch.sum(adv_labels != true_labels)

        attack_success_rate = successful_attacks.item() / len(true_labels)

        return attack_success_rate
