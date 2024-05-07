import click
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from advsecurenet.attacks import AdversarialAttack
from advsecurenet.models.base_model import BaseModel


def execute_attack(model: BaseModel,
                   data_loader: DataLoader,
                   attack: AdversarialAttack,
                   device: torch.device = torch.device("cpu"),
                   verbose: bool = False
                   ) -> list[torch.Tensor]:
    """
    Execute the specified attack on the model using the data loader.

    Args:
        model (BaseModel): The model to attack.
        data_loader (DataLoader): The data loader to use for generating adversarial samples.
        attack (AdversarialAttack): The attack to execute.
        device (torch.device): The device to use for the attack.
        verbose (bool): Whether to print verbose logs.

    Returns:
        list[torch.Tensor]: A list of adversarial images.

    """
    try:
        model = model.to(device)
        model.eval()
        adversarial_images = []

        successful_attacks = 0  # Track number of successful attacks
        total_samples = 0       # Track total number of samples processed

        for images, labels in tqdm(data_loader, desc="Generating adversarial samples"):
            images = images.to(device)
            labels = labels.to(device)

            # Get predictions for the original images
            original_preds = torch.argmax(model(images), dim=1)

            # Generate adversarial images
            adv_images = attack.attack(model, images, labels)
            adversarial_images.append(adv_images)

            # Get predictions for the adversarial images
            adversarial_preds = torch.argmax(model(adv_images), dim=1)

            # Check how many attacks were successful
            successful_attacks += (adversarial_preds !=
                                   original_preds).sum().item()
            total_samples += images.size(0)
            if verbose:
                click.echo(
                    f"Attack success rate: {successful_attacks / total_samples * 100:.2f}%")

        success_rate = (successful_attacks / total_samples) * 100
        print(
            f"Succesfully generated adversarial samples! Attack success rate: {success_rate:.2f}%")

        return adversarial_images
    except Exception as e:
        print("Error occurred while executing attack!")
        raise e
