import random

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from advsecurenet.attacks import AdversarialAttack
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

    def _combine_clean_and_adversarial_data(self, source: torch.Tensor, adv_source: torch.Tensor, targets: torch.Tensor, adv_targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # make sure that source and adv_source have the same shape and same normalization
        assert source.shape == adv_source.shape, "source and adv_source must have the same shape"

        # Combine clean and adversarial examples
        combined_data, combined_target = self._shuffle_data(
            torch.cat([source, adv_source], dim=0),
            torch.cat([targets, adv_targets], dim=0)
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
        source,
        targets,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly selects one model and one attack from the list of models and attacks to generate adversarial examples for the given batch.
        Args:
            source: A batch of clean images
            targets: A batch of clean labels
        """

        source, targets = self._move_to_device(source, targets)
        adv_source, adv_targets = [], []

        # Randomly select one model and one attack
        random_model = random.choice(self.config.models)
        random_attack = random.choice(self.config.attacks)

        # Move the model to the device
        random_model.to(self._device)

        # Set the model to eval mode
        random_model.eval()

        # Perform the attack using the unnormalized images
        attack_result = self._perform_attack(
            random_attack, random_model, source, targets
        )

        assert attack_result.shape == source.shape, "adversarial image and the clean image must have the same shape"

        adv_source.append(attack_result)
        adv_targets.append(targets)

        # Combine adversarial examples
        adv_source = torch.cat(adv_source, dim=0)
        adv_targets = torch.cat(adv_targets, dim=0)

        return adv_source, adv_targets

    def _move_to_device(self, source, targets):
        return source.to(self._device), targets.to(self._device)

    def _perform_attack(self,
                        attack: AdversarialAttack,
                        model: BaseModel,
                        source: torch.Tensor,
                        targets: torch.Tensor) -> torch.Tensor:
        """
        Performs the attack on the specified model and input.

        Args:
            attack (AdversarialAttack): The attack to perform.
            model (BaseModel): The model to attack.
            source (torch.tensor): The original input tensor. Expected shape is (batch_size, channels, height, width).
            targets (torch.tensor): The true labels for the input tensor. Expected shape is (batch_size,).

        Returns:
            torch.tensor: The adversarial example tensor.
        """
        if attack.name == "LOTS":
            paired = self.adversarial_target_generator.generate_target_images(
                zip(source, targets))
            original_images, _, target_images, target_labels = self.adversarial_target_generator.extract_images_and_labels(
                paired, source)
            # Perform attack
            adv_images, _ = attack.attack(  # type: ignore
                model=model,
                data=original_images,
                target=target_images,
                target_classes=target_labels,
            )
            return adv_images

        return attack.attack(model, source, targets)

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

        for _, (source, targets) in enumerate(train_loader):
            source, targets = self._prepare_data(source, targets)

            adv_source, adv_targets = self._generate_adversarial_batch(
                source, targets)
            adv_source, adv_targets = self._prepare_data(
                adv_source, adv_targets)

            combined_data, combined_targets = self._combine_clean_and_adversarial_data(
                source, adv_source, targets, adv_targets
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

    def _prepare_data(self, source, targets):
        source = source.to(self._device)
        targets = targets.to(self._device)
        return source, targets

    def _get_loss_divisor(self):
        return len(self.config.train_loader)
