import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import trange
from advsecurenet.utils import train, test
from advsecurenet.models.base_model import BaseModel
from advsecurenet.attacks import AdversarialAttack
from advsecurenet.shared.types import AdversarialTrainingConfig


def AdversarialTraining(config: AdversarialTrainingConfig) -> None:
    """
    This function implements the adversarial training defense. It allows to train a model using multiple models and attacks.

    Args:
        config (AdversarialTrainingConfig): The configuration for the adversarial training defense.

    Raises:
        ValueError: If the target_model is not a subclass of BaseModel.
        ValueError: If any of the models is not a subclass of BaseModel.
        ValueError: If any of the attacks is not a subclass of AdversarialAttack.
        ValueError: If the train_dataloader is not a DataLoader.


    """

    # First check if the model is subsclass of BaseModel
    if not isinstance(config.target_model, BaseModel):
        raise ValueError("Target model must be a subclass of BaseModel!")

    # Check if the models are subclass of BaseModel
    if not all([isinstance(model, BaseModel) for model in config.models]):
        raise ValueError("All models must be a subclass of BaseModel!")

    # Check if the attacks are subclass of AdversarialAttack
    if not all([isinstance(attack, AdversarialAttack) for attack in config.attacks]):
        raise ValueError(
            "All attacks must be a subclass of AdversarialAttack!")

    # Check if the train_dataloader is a DataLoader
    if not isinstance(config.train_dataloader, DataLoader):
        raise ValueError(
            "train_dataloader must be a torch.utils.data.DataLoader!")

    models = config.models
    attacks = config.attacks
    target_model = config.target_model
    optimizer = config.optimizer
    criterion = config.criterion
    epochs = config.epochs
    verbose = config.verbose
    device = config.device.value
    train_dataloader = config.train_dataloader

    for epoch in trange(epochs, desc="Epochs", leave=False, position=0, disable=not verbose):
        total_loss = 0.0

        # add target model to list of models if not already present
        if target_model not in models:
            models.append(target_model)

        # for batch_idx, (data, target) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs} ", leave=False, position=0, disable=not verbose)):
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)

            # generate adversarial examples
            adv_data = []
            adv_target = []
            for model, attack in zip(models, attacks):
                adv_data.append(attack.attack(model, data, target))
                adv_target.append(target)

            # concatenate adversarial examples
            adv_data = torch.cat(adv_data, dim=0)
            adv_target = torch.cat(adv_target, dim=0)

            # concatenate clean and adversarial examples
            data = torch.cat([data, adv_data], dim=0)
            target = torch.cat([target, adv_target], dim=0)

            # shuffle data
            permutation = torch.randperm(data.size(0))
            data = data[permutation]
            target = target[permutation]

            # train model
            optimizer.zero_grad()
            outputs = target_model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            break

        average_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch} - Average Loss: {average_loss:.6f}')
