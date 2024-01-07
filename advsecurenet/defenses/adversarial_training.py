import random

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from advsecurenet.attacks import AdversarialAttack
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import \
    AdversarialTrainingConfig
from advsecurenet.utils.adversarial_target_generator import \
    AdversarialTargetGenerator
from advsecurenet.utils.data import unnormalize_data
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
        self.scheduler = self._setup_scheduler()
        self.loss_fn = self._get_loss_function(self.config.criterion)
        self.start_epoch = self._load_checkpoint_if_any()
        self.adversarial_target_generator = AdversarialTargetGenerator()
        self.mean, self.std = self._get_meand_and_std()

    def _get_meand_and_std(self):
        # get mean and std of the dataset
        mean = self.config.train_loader.dataset.dataset.transform.transforms[-1].mean
        std = self.config.train_loader.dataset.dataset.transform.transforms[-1].std
        return mean, std

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

        # assert torch.allclose(source.mean(), adv_source.mean(
        # ), atol=1e-7), "source and adv_source must have the same mean"
        # assert torch.allclose(source.std(), adv_source.std(
        # ), atol=1e-7), "source and adv_source must have the same std"

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

        # first unnormalize the source. The operation is not in-place
        unnormalized_source = unnormalize_data(source, self.mean, self.std)

        # Perform the attack using the unnormalized images
        attack_result = self._perform_attack(
            random_attack, random_model, unnormalized_source, targets
        )

        assert attack_result.shape == source.shape, "adversarial image and the clean image must have the same shape"

        # normalize the adversarial examples to be in the same distribution as the clean examples
        attack_result = transforms.Normalize(
            mean=self.mean, std=self.std)(attack_result)

        adv_source.append(attack_result)
        adv_targets.append(targets)

        # Combine adversarial examples
        adv_source = torch.cat(adv_source, dim=0)
        adv_targets = torch.cat(adv_targets, dim=0)

        return adv_source, adv_targets

    def _move_to_device(self, source, targets):
        return source.to(self.device), targets.to(self.device)

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

        total_loss = 0.0
        for _, (source, targets) in enumerate(tqdm(self.config.train_loader, desc="Adversarial Training",
                                                   leave=False, position=1, unit="batch", colour="blue")):

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

        self._log_loss(epoch, total_loss)
