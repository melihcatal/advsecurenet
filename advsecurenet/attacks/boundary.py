from collections import deque
from typing import Optional

import click
import torch
from tqdm.auto import trange

import advsecurenet.shared.types.configs.attack_configs as AttackConfigs
from advsecurenet.attacks.adversarial_attack import AdversarialAttack
from advsecurenet.models.base_model import BaseModel


class DecisionBoundary(AdversarialAttack):
    """
    Decision Boundary attack

    Paper: https://arxiv.org/abs/1712.04248
    """

    def __init__(self, config: AttackConfigs.DecisionBoundaryAttackConfig) -> None:
        self.initial_delta = config.initial_delta
        self.initial_epsilon = config.initial_epsilon
        self.max_delta_trials = config.max_delta_trials
        self.max_epsilon_trials = config.max_epsilon_trials
        self.max_iterations = config.max_iterations
        self.max_initialization_trials = config.max_initialization_trials
        self.step_adapt = config.step_adapt
        self.targeted = config.targeted
        self.verbose = config.verbose
        self.early_stopping = config.early_stopping
        self.early_stopping_threshold = config.early_stopping_threshold
        self.early_stopping_patience = config.early_stopping_patience
        super().__init__(config)

    def _initialize(self, model, original_images, true_labels, target_labels=None):
        """
        Initializes the perturbed images for the Decision Boundary attack. The initialization is done by randomly and trying to find an adversarial example in given number of iterations.
        If the attack is targeted, initialization tries to find an adversarial example that is classified as the target label. If the attack is untargeted, initialization tries to find an adversarial example that is misclassified (i.e. not classified as the true label).
        """
        model.eval()

        # Ensure original_images, true_labels, and target_labels are all tensors on the same device
        original_images = self.device_manager.to_device(original_images)
        true_labels = self.device_manager.to_device(true_labels)

        if target_labels is not None:
            target_labels = self.device_manager.to_device(target_labels)

        # Create a mask for images that need to be updated, initially all are True
        update_mask = torch.ones(original_images.size(
            0), dtype=torch.bool, device=original_images.device)

        perturbed_images = original_images.clone()

        for _ in trange(self.max_initialization_trials, desc="Initializing", colour="yellow", disable=not self.verbose):
            # Randomly initialize the perturbed images for those that need to be updated
            random_imgs = torch.rand_like(original_images)
            perturbed_images[update_mask] = random_imgs[update_mask]

            # Forward pass with the perturbed images
            outputs = model(perturbed_images)
            preds = outputs.argmax(dim=1)

            # Update the mask based on whether the attack is targeted or untargeted
            if self.targeted:
                assert target_labels is not None, "Target labels must be provided for a targeted attack"
                # Ensure this is a boolean tensor
                update_mask = ~(preds == target_labels)
            else:
                update_mask = preds == true_labels  # Ensure this is a boolean tensor

            # Check and debug the type of update_mask
            assert update_mask.dtype == torch.bool, "Update mask must be a boolean tensor"
            # Check if all images are adversarial according to the respective condition
            if not update_mask.any():
                if self.verbose:
                    click.echo(click.style(
                        "Successfully initialized all images", fg="green"))
                break

        if update_mask.all() and self.verbose:
            click.echo(click.style(
                "Initialization failed. Try increasing max_initialization_trials", fg="red"))

        # only return the perturbed images that successfully initialized
        # return perturbed_images[update_mask]

        return perturbed_images

    def _orthogonal_perturb(self, delta, current_samples, original_samples):
        batch_size, channels, height, width = current_samples.shape
        # Generate perturbation randomly for a batch of images
        perturb = torch.randn_like(current_samples)

        # Rescale the perturbation
        perturb_norm = torch.norm(perturb.view(
            batch_size, -1), dim=1, keepdim=True)
        perturb = perturb / perturb_norm.view(batch_size, 1, 1, 1)

        diff = original_samples - current_samples
        diff_norm = torch.norm(diff.view(batch_size, -1), dim=1, keepdim=True)
        perturb = perturb * (delta * diff_norm.view(batch_size, 1, 1, 1))

        # Project the perturbation onto the sphere
        direction = diff / diff_norm.view(batch_size, 1, 1, 1)
        perturb_flat = perturb.view(batch_size, -1)
        direction_flat = direction.view(batch_size, -1)

        # Remove component in the direction of (original - current)
        adjustment = torch.bmm(perturb_flat.unsqueeze(
            1), direction_flat.unsqueeze(2)).squeeze(2)
        perturb_flat -= adjustment * direction_flat
        perturb = perturb_flat.view_as(perturb)

        # Calculate the final perturbed image
        delta = torch.tensor(delta, dtype=torch.float32,
                             device=current_samples.device)
        hypotenuse = torch.sqrt(1 + delta ** 2)
        perturb = ((1 - hypotenuse) * diff + perturb) / hypotenuse

        return perturb

    def _forward_perturb(self, epsilon, adv_images, original_images):
        """
        Generates a perturbation in the direction of the original image.

        Args:
            epsilon (float): The epsilon value to use for the attack.
            adv_images (torch.tensor): The adversarial images. Expected shape is (batch_size, channels, height, width).
            original_images (torch.tensor): The original images. Expected shape is (batch_size, channels, height, width).

        Returns:

            torch.tensor: The perturbation tensor.
        """
        # Calculate the direction vector from the adversarial images towards the original images
        direction = original_images - adv_images
        # Calculate the norm of the direction (batch-wise)
        norm = torch.norm(direction.view(direction.size(0), -1),
                          p=2, dim=1).view(-1, 1, 1, 1)
        # Avoid division by zero
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        # Calculate the perturbation
        perturbation = epsilon * direction / norm

        return perturbation

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses the images before attack. The attack expects the images to be in the range [0, 1] and normalized with the mean and standard deviation of the dataset.
        The datasets are already normalized with the mean and standard deviation of the dataset, so we only need to check if the images are in the range [0, 1].

        Args:
            images (torch.Tensor): The images to preprocess.

        Returns:
            torch.Tensor: The preprocessed images.
        """

        if images.min() < 0 or images.max() > 1:
            if self.verbose:
                click.echo(click.style(
                    "Images are not in the range [0, 1], normalizing them...", fg="red"))
            # The images are not in the range [0, 1], so we need to normalize them
            images = torch.clamp(images, min=0, max=1)

        return images

    # mypy: ignore-errors
    def attack(self, model: BaseModel, original_images: torch.Tensor, true_labels: torch.Tensor, target_labels: Optional[torch.Tensor] = None, * args, **kwargs) -> torch.Tensor:
        """
        Generates adversarial examples using the Decision Boundary attack.


        """
        if self.targeted and target_labels is None:
            raise ValueError(
                "Target labels must be provided for a targeted attack")

        if self.targeted and target_labels is not None:
            target_labels = self.device_manager.to_device(target_labels)

        model.eval()
        batch_size = original_images.size(0)

        # preprocess the images
        original_images = self._preprocess(original_images)

        adv_images = self._initialize(
            model, original_images, true_labels, target_labels)

        adv_images = self.device_manager.to_device(adv_images)
        best_adv_images = adv_images.clone()
        best_distances = torch.full((batch_size,), float(
            'inf'), device=original_images.device)

        # copy the initial delta and epsilon values
        delta = self.initial_delta
        epsilon = self.initial_epsilon

        # Initialize a list or deque to track the last 'n' improvements in best distance, where 'n' is the patience. Used only if early stopping is enabled.
        recent_improvements = deque(maxlen=self.early_stopping_patience)

        # try to get closer to the decision boundary
        for iteration in trange(self.max_iterations, desc="Boundary Attack", colour="blue", disable=not self.verbose):

            # move orthogonal to the decision boundary
            for _ in range(self.max_delta_trials):
                perturbation = self._orthogonal_perturb(
                    delta, adv_images, original_images)
                trial_images = adv_images + perturbation
                trial_images.clamp_(min=0, max=1)

                outputs = model(trial_images)
                predictions = outputs.argmax(dim=1)
                if self.targeted:
                    success = predictions == target_labels
                else:
                    success = predictions != true_labels

                success_rate = success.float().mean().item()
                if success_rate < 0.2:
                    delta *= self.step_adapt
                elif success_rate > 0.5:
                    delta /= self.step_adapt

                for idx in range(batch_size):
                    if success[idx]:
                        adv_images[idx] = trial_images[idx]

            # move towards the original image
            for _ in range(self.max_epsilon_trials):
                perturbation = self._forward_perturb(
                    epsilon, adv_images, original_images)
                trial_images = adv_images + perturbation
                trial_images = torch.clamp(trial_images, min=0, max=1)

                outputs = model(trial_images)
                predictions = outputs.argmax(dim=1)

                if self.targeted:
                    success = predictions == target_labels
                else:
                    success = predictions != true_labels

                success_rate = success.float().mean().item()
                if success_rate < 0.2:
                    epsilon *= self.step_adapt
                elif success_rate > 0.5:
                    epsilon /= self.step_adapt

                for idx in range(batch_size):
                    if success[idx]:
                        adv_images[idx] = trial_images[idx]

            with torch.no_grad():
                distances = torch.norm(
                    adv_images - original_images, p=2, dim=(1, 2, 3))
                improved = distances < best_distances
                best_adv_images[improved] = adv_images[improved]
                best_distances[improved] = distances[improved]

                if self.early_stopping:
                    # Update the recent improvements tracker
                    if iteration >= self.early_stopping_patience:
                        # Calculate improvement over the last 'patience' iterations
                        improvement = recent_improvements[0] - \
                            best_distances.mean().item()
                        recent_improvements.append(
                            best_distances.mean().item())

                        # Check if the improvement is less than the threshold for early stopping
                        if improvement < self.early_stopping_threshold:
                            if self.verbose:
                                click.echo(click.style(
                                    "Early stopping", fg="yellow"))
                            break
                    else:
                        # Just append the current mean distance before the patience threshold is reached
                        recent_improvements.append(
                            best_distances.mean().item())

        return best_adv_images
