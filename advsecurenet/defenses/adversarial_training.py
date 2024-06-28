import random
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from advsecurenet.attacks import AdversarialAttack
from advsecurenet.datasets.targeted_adv_dataset import AdversarialDataset
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import \
    AdversarialTrainingConfig
from advsecurenet.trainer.trainer import Trainer
from advsecurenet.utils.adversarial_target_generator import \
    AdversarialTargetGenerator


class AdversarialTraining(Trainer):
    """
    Adversarial Training class. This module implements the Adversarial Training defense.

    Args:
        config (AdversarialTrainingConfig): The configuration for the Adversarial Training defense.

    """

    def __init__(self, config: AdversarialTrainingConfig) -> None:
        self._check_config(config)
        self.config: AdversarialTrainingConfig = config
        self.adversarial_target_generator = AdversarialTargetGenerator()
        super().__init__(config)

    # Helper function to shuffle the combined clean and adversarial data

    def _shuffle_data(self, data: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        permutation = torch.randperm(data.size(0))
        return data[permutation], target[permutation]

    def _check_config(self, config: AdversarialTrainingConfig) -> None:
        # Check configuration validity
        if not isinstance(config.model, BaseModel):
            raise ValueError("Target model must be a subclass of BaseModel!")
        if not all(isinstance(model, BaseModel) for model in config.models):
            raise ValueError("All models must be a subclass of BaseModel!")
        if not all(isinstance(attack, AdversarialAttack) for attack in config.attacks):
            raise ValueError(
                "All attacks must be a subclass of AdversarialAttack!")
        if not isinstance(config.train_loader, DataLoader):
            raise ValueError("train_dataloader must be a DataLoader!")

        # check if any of the attacks are targeted and if so, check if the dataloader dataset is an instance of AdversarialDataset
        if any(attack.targeted for attack in config.attacks) and not isinstance(config.train_loader.dataset, AdversarialDataset):
            raise ValueError(
                "If any of the attacks are targeted, the train_loader dataset must be an instance of AdversarialDataset!"
            )
        # if any of the attacks is LOTS, check if the dataset contains target images and target labels
        if any(attack.name == "LOTS" for attack in config.attacks) and not isinstance(config.train_loader.dataset, AdversarialDataset) and len(config.train_loader) != 4:
            raise ValueError(
                "If the LOTS attack is used, the train_loader dataset must be an instance of AdversarialDataset and must contain target images and target labels!"
            )

    def _combine_clean_and_adversarial_data(self, images: torch.Tensor, adv_source: torch.Tensor, true_labels: torch.Tensor, adv_targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # make sure that images and adv_source have the same shape and same normalization
        assert images.shape == adv_source.shape, "images and adv_source must have the same shape"

        # Combine clean and adversarial examples
        combined_data, combined_target = self._shuffle_data(
            torch.cat([images, adv_source], dim=0),
            torch.cat([true_labels, adv_targets], dim=0)
        )
        return combined_data, combined_target

    def _pre_training(self):
        # add target model to list of models if not already present
        if self.config.model not in self.config.models:
            self.config.models.append(self.config.model)

        # set each model to train mode
        self.config.models = [model.train() for model in self.config.models]

        # move each model to device
        self.config.models = [model.to(self._device)
                              for model in self.config.models]

    def _generate_adversarial_batch(
        self,
        images,
        true_labels,
        target_images: Optional[torch.Tensor] = None,
        target_labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly selects one model and one attack from the list of models and attacks to generate adversarial examples for the given batch.
        Args:
            images: A batch of clean images
            true_labels: A batch of clean labels
            target_images: A batch of target images
            target_labels: A batch of target labels
        """

        images, true_labels = self._move_to_device(images, true_labels)
        adv_source, adv_targets = [], []

        # Randomly select one model and one attack
        random_model = random.choice(self.config.models)
        random_attack = random.choice(self.config.attacks)

        # Move the model to the device
        random_model.to(self._device)

        # Set the model to eval mode
        random_model.eval()

        attack_result = self._perform_attack(
            random_attack, random_model, images, true_labels, target_images, target_labels
        )

        assert attack_result.shape == images.shape, "adversarial image and the clean image must have the same shape"

        adv_source.append(attack_result)
        adv_targets.append(true_labels)

        # Combine adversarial examples
        adv_source = torch.cat(adv_source, dim=0)
        adv_targets = torch.cat(adv_targets, dim=0)

        return adv_source, adv_targets

    def _move_to_device(self, images, true_labels):
        return images.to(self._device), true_labels.to(self._device)

    def _perform_attack(self,
                        attack: AdversarialAttack,
                        model: BaseModel,
                        images: torch.Tensor,
                        true_labels: torch.Tensor,
                        target_images: Optional[torch.Tensor] = None,
                        target_labels: Optional[torch.Tensor] = None
                        ) -> torch.Tensor:
        """
        Performs the attack on the specified model and input.

        Args:
            attack (AdversarialAttack): The attack to perform.
            model (BaseModel): The model to attack.
            images (torch.tensor): The original input tensor. Expected shape is (batch_size, channels, height, width).
            true_labels (torch.tensor): The true labels for the input tensor. Expected shape is (batch_size,).
            target_images (Optional[torch.tensor], optional): The target input tensor. Expected shape is (batch_size, channels, height, width). Defaults to None.
            target_labels (Optional[torch.tensor], optional): The target labels for the target input tensor. Expected shape is (batch_size,). Defaults to None.

        Returns:
            torch.tensor: The adversarial example tensor.
        """
        if attack.targeted:
            if attack.name == "LOTS":
                # if the attack is LOTS we need to provide target images as well
                return attack.attack(model, images, target_labels, target_images)

            # if the attack is targeted we provide target labels
            return attack.attack(model, images, target_labels)

        # the attack is untargeted
        return attack.attack(model, images, true_labels)

    def _run_epoch(self, epoch: int) -> None:
        """
        Run a single epoch of adversarial training.

        Args:
            epoch (int): The current epoch number.

        Returns:
            None
        """
        total_loss = 0.0
        train_loader = self._get_train_loader(epoch)
        for data in train_loader:
            if len(data) == 4:
                images, true_labels, target_images, target_labels = data
                images, true_labels, target_images, target_labels = self._prepare_data(
                    images, true_labels, target_images, target_labels)

            else:
                images, true_labels = data
                target_images = None
                target_labels = None
                images, true_labels = self._prepare_data(images, true_labels)

            adv_source, adv_targets = self._generate_adversarial_batch(
                images,
                true_labels,
                target_images,
                target_labels

            )
            adv_source, adv_targets = self._prepare_data(
                adv_source, adv_targets)

            combined_data, combined_targets = self._combine_clean_and_adversarial_data(
                images, adv_source, true_labels, adv_targets
            )

            loss = self._run_batch(combined_data, combined_targets)
            total_loss += loss

        total_loss /= self._get_loss_divisor()
        self._log_loss(epoch, total_loss)

    def _get_train_loader(self, epoch: int):
        return tqdm(self.config.train_loader,
                    desc="Adversarial Training",
                    leave=False,
                    position=1,
                    unit="batch",
                    colour="blue")

    def _prepare_data(self, *args):
        """
        Move the required data to the device.
        """
        return [arg.to(self._device) for arg in args if arg is not None]

    def _get_loss_divisor(self):
        return len(self.config.train_loader)
