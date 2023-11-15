import os
import torch
from torch import nn, optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from advsecurenet.models.base_model import BaseModel
from advsecurenet.attacks import AdversarialAttack
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import AdversarialTrainingConfig


class AdversarialTraining:
    """
    Adversarial Training class.
    """

    def __init__(self, config: AdversarialTrainingConfig) -> None:
        self.config = config

    def adversarial_training(self) -> None:
        # Check configuration validity
        self._check_config(self.config)
        device = _setup_device(self.config)
        optimizer = _initialize_optimizer(self.config)
        print(f"Adversarial Training: Using {device} for training")
        self._adversarial_training(self.config, device, optimizer)

    # Helper function to generate adversarial examples for the given batch
    def _generate_adversarial_batch(self, models, attacks, data, target, device) -> tuple[torch.Tensor, torch.Tensor]:
        adv_data = []
        adv_target = []
        data = data.to(device)
        target = target.to(device)
        for model, attack in zip(models, attacks):
            model.to(device)
            adv_data.append(attack.attack(model, data, target))
            adv_target.append(target)
        return torch.cat(adv_data, dim=0), torch.cat(adv_target, dim=0)

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

    def _train_epoch(self, config: AdversarialTrainingConfig, device: torch.device, optimizer: optim.Optimizer, loss_function: nn.Module, epoch: int) -> float:
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(config.train_loader, desc=f"Epoch {epoch}/{config.epochs}", total=len(config.train_loader))):
            # generate adversarial examples
            adv_data, adv_target = self._generate_adversarial_batch(
                config.models, config.attacks, data, target, device)

            data = data.to(device)
            target = target.to(device)
            adv_data = adv_data.to(device)
            adv_target = adv_target.to(device)

            # Combine clean and adversarial examples
            combined_data, combined_target = self._shuffle_data(
                torch.cat([data, adv_data], dim=0),
                torch.cat([target, adv_target], dim=0)
            )

            # train model
            optimizer.zero_grad()
            outputs = config.model(combined_data)
            loss = loss_function(outputs, combined_target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss

    def _adversarial_training(self, config: AdversarialTrainingConfig, device: torch.device, optimizer: optim.Optimizer) -> None:
        # set each model to device
        config.models = [model.to(device) for model in config.models]

        config.model = config.model.to(device)

        # add target model to list of models if not already present
        if config.model not in config.models:
            config.models.append(config.model)

        # set each model to train mode
        config.models = [model.train() for model in config.models]

        # initalize loss function
        loss_function = _get_loss_function(config.criterion)
        start_epoch = _load_checkpoint_if_any(config, device, optimizer)

        # train model
        for epoch in range(start_epoch, config.epochs + 1):
            total_loss = self._train_epoch(
                config, device, optimizer, loss_function, epoch)
            average_loss = total_loss / len(config.train_loader)
            if config.verbose:
                print(f'Epoch {epoch} - Average Loss: {average_loss:.6f}')

            # Save checkpoint if applicable
            if config.save_checkpoint and epoch % config.checkpoint_interval == 0:
                _save_checkpoint(config, epoch, optimizer=optimizer)

        print("Adversarial Training: Training complete!")
