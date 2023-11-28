import copy
import csv
import os
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
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

    def calculate_l0_distance(self, original_images: torch.Tensor, adversarial_images: torch.Tensor) -> float:
        """
        Calculates the L0 distance between the original and adversarial images. L0 distance is the count of pixels that are different between the two images (i.e. the number of pixels that have been changed in the adversarial image compared to the original image).

        Args:
            original_images (torch.Tensor): The original images.
            adversarial_images (torch.Tensor): The adversarial images.

        Returns:
            float: The mean L0 distance between the original and adversarial images.
        """

        # Calculating the L0 distance
        l0_distance = (original_images !=
                       adversarial_images).sum(dim=(1, 2, 3))

        # Convert to floating point before taking the mean
        l0_distance = l0_distance.float().mean()

        return l0_distance.item()

    def calculate_l2_distance(self, original_images: torch.Tensor, adversarial_images: torch.Tensor) -> float:
        """ 
        Calculates the L2 distance between the original and adversarial images. L2 distance is the Euclidean distance between the two images.

        Args:
            original_images (torch.Tensor): The original images.
            adversarial_images (torch.Tensor): The adversarial images.

        Returns:
            float: The mean L2 distance between the original and adversarial images.
        """
        l2_distance = torch.norm(
            (original_images - adversarial_images).view(original_images.shape[0], -1), p=2, dim=1).mean()
        return l2_distance.item()

    def calculate_l_inf_distance(self, original_images: torch.Tensor, adversarial_images: torch.Tensor) -> float:
        """
        Calculates the L∞ distance between the original and adversarial images. L∞ distance is the maximum absolute difference between the two images in any pixel.

        Args:
            original_images (torch.Tensor): The original images.
            adversarial_images (torch.Tensor): The adversarial images.

        Returns:
            float: The mean L∞ distance between the original and adversarial images.
        """
        l_inf_distance = (original_images - adversarial_images).view(
            original_images.shape[0], -1).abs().max(dim=1)[0].mean()
        return l_inf_distance.item()

    def calculate_perturbation_distances(self, original_images: torch.Tensor, adversarial_images: torch.Tensor) -> tuple[float, float, float]:
        """
        Calculates the L0, L2, and L∞ distances between the original and adversarial images. 

        Args:
            original_images (torch.Tensor): The original images.
            adversarial_images (torch.Tensor): The adversarial images.

        Returns:
            Tuple[float, float, float]: The mean L0, L2, and L∞ distances between the original and adversarial images.
        """
        l0_distance = self.calculate_l0_distance(
            original_images, adversarial_images)
        l2_distance = self.calculate_l2_distance(
            original_images, adversarial_images)
        l_inf_distance = self.calculate_l_inf_distance(
            original_images, adversarial_images)
        return l0_distance, l2_distance, l_inf_distance

    def calculate_perturbation_effectiveness_score(self, attack_success_rate: float, perturbation_distance: float) -> float:
        """ 
        Calculates the perturbation effectiveness score for the attack. The effectiveness score is the attack success rate divided by the perturbation distance. The higher the score, the more effective the attack. 
        The purpose of this metric is to distinguish between attacks that have a high success rate but require a large perturbation magnitude, 
        and attacks that have a lower success rate but require a smaller perturbation magnitude.
        Args:
            attack_success_rate (float): The attack success rate.
            perturbation_distance (float): The perturbation distance.

        Returns:
            float: The effectiveness score.
        """

        return attack_success_rate / perturbation_distance

    def calculate_robustness_gap(self, model: BaseModel, original_images: torch.Tensor, true_labels: torch.Tensor, adversarial_images: torch.Tensor) -> float:
        """
        Calculates the robustness gap for the adversarial examples. The robustness gap is the difference between the accuracy of the model on the original images and the accuracy of the model on the adversarial images.
        The larger the robustness gap, the more effective the attack.

        Args:
            original_images (torch.Tensor): The original images.
            true_labels (torch.Tensor): The true labels for the original images.
            adversarial_images (torch.Tensor): The adversarial images.

        Returns:
            float: The mean robustness gap for the adversarial examples.
        """
        clean_accuracy = self._calculate_accuracy(
            model, original_images, true_labels)
        # TODO: Better naming here
        adversarial_accuracy = self._calculate_accuracy(
            model, adversarial_images, true_labels)
        return clean_accuracy - adversarial_accuracy

    def calculate_ssim(self, original_images: torch.Tensor, adversarial_images: torch.Tensor) -> float:
        """
        Calculates the mean structural similarity index (SSIM) between the original and adversarial images. SSIM is a metric that measures the similarity between two images. 
        The higher the SSIM, the more similar the images are. 

        Args:
            original_images (torch.Tensor): The original images. Expected shape is (batch_size, channels, height, width).
            adversarial_images (torch.Tensor): The adversarial images.

        Returns:
            float: The mean SSIM between the original and adversarial images. [-1, 1] range. 1 means the images are identical.
        """
        # Convert tensors to numpy arrays and ensure they are float32
        original_images_np = original_images.cpu().detach().numpy().astype(np.float32)
        adversarial_images_np = adversarial_images.cpu().detach().numpy().astype(np.float32)
        data_range = original_images_np.max() - original_images_np.min()
        ssim_score = ssim(original_images_np,
                          adversarial_images_np,
                          channel_axis=1,
                          data_range=data_range,
                          )

        return ssim_score.mean()

    def calculate_psnr(self, original_images: torch.Tensor, adversarial_images: torch.Tensor) -> float:
        """
        Calculates the mean peak signal-to-noise ratio (PSNR) between the original and adversarial images. PSNR is a metric that measures the similarity between two images. 
        The higher the PSNR, the more similar the images are and the lower the distortion between them. A high PSNR could indicate that the perturbations introduced are
        subtle but may not necessarily reflect the perceptual similarity between the images.

        Args:
            original_images (torch.Tensor): The original images. Expected shape is (batch_size, channels, height, width).
            adversarial_images (torch.Tensor): The adversarial images.

        Returns:
            float: The mean PSNR between the original and adversarial images. [0, inf) range. Higher values (e.g. 30 dB or more) indicate better quality. Infinite if the images are identical.
        """
        original_images_np = original_images.cpu().detach().numpy().astype(np.float32)
        adversarial_images_np = adversarial_images.cpu().detach().numpy().astype(np.float32)
        data_range = original_images_np.max() - original_images_np.min()
        psnr_score = psnr(original_images_np,
                          adversarial_images_np, data_range=data_range)
        return psnr_score.mean().item()

    def calculate_similarity_scores(self, original_images: torch.Tensor, adversarial_images: torch.Tensor) -> tuple[float, float]:
        """
        Calculates the SSIM and PSNR between the original and adversarial images. 

        Args:
            original_images (torch.Tensor): The original images. Expected shape is (batch_size, channels, height, width).
            adversarial_images (torch.Tensor): The adversarial images.

        Returns:
            Tuple[float, float]: The mean SSIM and PSNR between the original and adversarial images.
        """
        ssim_score = self.calculate_ssim(original_images, adversarial_images)
        psnr_score = self.calculate_psnr(original_images, adversarial_images)
        return ssim_score, psnr_score

    def calculate_transferability_rate(self,
                                       source_model,
                                       target_model,
                                       original_images: torch.Tensor,
                                       true_labels: torch.Tensor,
                                       adversarial_images: torch.Tensor,
                                       is_targeted: bool = False,
                                       target_labels: Optional[torch.Tensor] = None) -> float:
        """
        Calculates the attack transferability rate. This rate is the percentage of adversarial examples that are successful on the target model among those that were successful on the source model.

        Args:
            source_model: The model used to generate adversarial examples.
            target_model: A different model to test the transferability of adversarial examples.
            original_images (torch.Tensor): The original images.
            true_labels (torch.Tensor): The true labels for the original images.
            adversarial_images (torch.Tensor): The adversarial images generated from the source model.
            is_targeted (bool, optional): Whether the attack is targeted.
            target_labels (Optional[torch.Tensor], optional): The target labels for the adversarial images if the attack is targeted.

        Returns:
            float: The attack transferability rate.
        """
        source_model.eval()
        target_model.eval()

        # Get the predictions for adversarial images on both models
        source_predictions = source_model(adversarial_images)
        target_predictions = target_model(adversarial_images)

        # Convert predictions to labels
        source_labels = torch.argmax(source_predictions, dim=1)
        target_labels = torch.argmax(target_predictions, dim=1)

        if is_targeted:
            if target_labels is None:
                raise ValueError(
                    "Target labels must be provided for targeted attacks.")
            # Count how many adversarial examples were successful on the source model
            successful_on_source = torch.sum(source_labels == target_labels)
            # Count how many of these are also successful on the target model
            successful_transfer = torch.sum(
                (source_labels == target_labels) & (target_labels == true_labels))
        else:
            # Count how many adversarial examples were successful on the source model
            successful_on_source = torch.sum(source_labels != true_labels)
            # Count how many of these are also successful on the target model
            successful_transfer = torch.sum(
                (source_labels != true_labels) & (target_labels != true_labels))

        # Calculate transferability rate
        transferability_rate = (successful_transfer.float(
        ) / successful_on_source.float()) if successful_on_source > 0 else 0

        # Return transferability rate, converting to Python float if it's a tensor
        return transferability_rate.item() if successful_on_source > 0 else transferability_rate

    def save_results_to_csv(self,
                            evaluation_results: dict,
                            experiment_info: Optional[dict] = None,
                            path: Optional[str] = None,
                            file_name: Optional[str] = None
                            ) -> None:
        """
        Saves the evaluation results to a CSV file in a structured format.

        Args:
            evaluation_results (dict): The evaluation results.
            experiment_info (dict, optional): The experiment info.
            path (str, optional): The path where the file will be saved.
            file_name (str, optional): The name of the CSV file.

        """

        # Create file name with timestamp if not provided
        if file_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f"{timestamp}_experiment.csv"

        # Create path if provided and does not exist
        if path:
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(path, file_name)
        else:
            file_path = file_name

        file_exists = os.path.exists(file_path)

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Write the experiment info and headers if file doesn't exist
            if not file_exists and experiment_info is not None:
                writer.writerow(
                    [f"Experiment conducted on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
                for key, value in experiment_info.items():
                    writer.writerow([key, value])

                # Write a separator
                writer.writerow(["-" * 10, "-" * 10])

                # Write headers
                headers = list(evaluation_results.keys())
                writer.writerow(headers)

            # Write the actual results
            values = [str(value)
                      for value in evaluation_results.values()]
            writer.writerow(values)

    def _calculate_accuracy(self, model: BaseModel, images: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Calculates the accuracy of the model on the given images.
        """
        model.eval()
        predictions = model(images)
        predicted_labels = torch.argmax(predictions, dim=1)
        correct_predictions = torch.sum(predicted_labels == labels)
        accuracy = correct_predictions.item() / len(labels)
        return accuracy
