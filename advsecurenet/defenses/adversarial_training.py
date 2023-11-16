import os
import torch
from torch import nn, optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from advsecurenet.models.base_model import BaseModel
from advsecurenet.attacks import AdversarialAttack
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import AdversarialTrainingConfig
from advsecurenet.utils.trainer import Trainer


class AdversarialTraining(Trainer):
    """
    Adversarial Training class.
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

    # Helper function to generate adversarial examples for the given batch
    def _generate_adversarial_batch(self, source, targets) -> tuple[torch.Tensor, torch.Tensor]:
        adv_source = []
        adv_targets = []
        source = source.to(self.device)
        targets = targets.to(self.device)
        for model, attack in zip(self.config.models, self.config.attacks):
            model.to(self.device)
            adv_source.append(attack.attack(model, source, targets))
            adv_targets.append(targets)

        return torch.cat(adv_source, dim=0), torch.cat(adv_targets, dim=0)

    def _run_epoch(self, epoch: int) -> None:
        total_loss = 0.0
        for batch_idx, (source, targets) in enumerate(tqdm(self.config.train_loader, desc=f"Epoch {epoch}/{self.config.epochs}", total=len(self.config.train_loader))):
            # generate adversarial examples
            adv_source, adv_targets = self._generate_adversarial_batch(
                source=source, targets=targets)

            source = source.to(self.device)
            targets = targets.to(self.device)
            adv_source = adv_source.to(self.device)
            adv_targets = adv_targets.to(self.device)

            combined_data, combined_target = self._combine_clean_and_adversarial_data(
                source=source, adv_source=adv_source, targets=targets, adv_targets=adv_targets)

            loss = self._run_batch(combined_data, combined_target)
            total_loss += loss
        total_loss /= len(self.config.train_loader)
        print(f"Epoch {epoch}/{self.config.epochs} Loss: {total_loss}")
