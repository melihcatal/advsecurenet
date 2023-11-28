import random

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from advsecurenet.attacks import AdversarialAttack
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import \
    AdversarialTrainingConfig
from advsecurenet.utils.adversarial_target_generator import \
    AdversarialTargetGenerator
from advsecurenet.utils.trainer import Trainer


class AdversarialTraining(Trainer):
    """
    Adversarial Training class. This module implements the Adversarial Training defense.

    Args:
        config (AdversarialTrainingConfig): The configuration for the Adversarial Training defense.

    """

    def __init__(self, config: AdversarialTrainingConfig) -> None:
        # first check the config
        self._check_config(config)
        self.config: AdversarialTrainingConfig = config
        self.device = self._setup_device()
        self.model = self._setup_model()
        self.optimizer = self._setup_optimizer()
        self.loss_fn = self._get_loss_function(self.config.criterion)
        self.start_epoch = self._load_checkpoint_if_any()
        self.adversarial_target_generator = AdversarialTargetGenerator()

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
        self.config.models = [model.to(self.device)
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
            batch_idx: The index of the batch
            lots_source: A batch of LOTS source images
            lots_targets: A batch of LOTS target images
        """

        source, targets = self._move_to_device(source, targets)
        adv_source, adv_targets = [], []

        # Randomly select one model and one attack
        random_model = random.choice(self.config.models)
        random_attack = random.choice(self.config.attacks)

        # Move the model to the device
        random_model.to(self.device)

        # Set the model to eval mode
        random_model.eval()

        # Perform the attack
        attack_result = self._perform_attack(
            random_attack, random_model, source, targets
        )

        adv_source.append(attack_result)
        adv_targets.append(targets)

        return torch.cat(adv_source, dim=0), torch.cat(adv_targets, dim=0)

    def _move_to_device(self, source, targets):
        return source.to(self.device), targets.to(self.device)

    def _perform_attack(self, attack, model, source, targets):
        if attack.name == "LOTS":
            paired = self.adversarial_target_generator.generate_target_images(
                zip(source, targets))
            original_images, _, target_images, target_labels = self.adversarial_target_generator.extract_images_and_labels(
                paired, source)
            # Perform attack
            adv_images, _ = attack.attack(
                model=model,
                data=original_images,
                target=target_images,
                target_classes=target_labels,
            )
            return adv_images
        else:
            return attack.attack(model, source, targets)

    def _run_epoch(self, epoch: int) -> None:

        print(f"Running epoch {epoch}...")

        total_loss = 0.0
        for batch_idx, (source, targets) in enumerate(tqdm(self.config.train_loader)):

            # Move data to device
            source = source.to(self.device)
            targets = targets.to(self.device)

            # Generate adversarial examples
            adv_source, adv_targets = self._generate_adversarial_batch(
                source=source,
                targets=targets
            )

            # Move adversarial examples to device
            adv_source = adv_source.to(self.device)
            adv_targets = adv_targets.to(self.device)

            # Combine clean and adversarial examples
            combined_data, combined_targets = self._combine_clean_and_adversarial_data(
                source=source,
                adv_source=adv_source,
                targets=targets,
                adv_targets=adv_targets
            )
            loss = self._run_batch(combined_data, combined_targets)
            total_loss += loss

        # Compute average loss across all batches and all processes
        total_loss /= len(self.config.train_loader)

        print(f"Epoch {epoch}/{self.config.epochs} Loss: {total_loss}")
