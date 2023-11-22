import torch
import click
from tqdm import tqdm
from advsecurenet.models.base_model import BaseModel
from advsecurenet.datasets.base_dataset import BaseDataset
from advsecurenet.attacks import AdversarialAttack
from advsecurenet.dataloader import DataLoaderFactory


def execute_attack(model: BaseModel,
                   data: BaseDataset,
                   batch_size: int,
                   attack: AdversarialAttack,
                   device: torch.device = torch.device("cpu"),
                   verbose: bool = False
                   ) -> list[torch.Tensor]:
    try:
        data_loader = DataLoaderFactory.create_dataloader(
            data, batch_size=batch_size)

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
