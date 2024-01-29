import torch
from tqdm.auto import tqdm

from advsecurenet.attacks import AdversarialAttack
from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.models.base_model import BaseModel


def execute_attack(model: BaseModel,
                   data_loader: DataLoaderFactory,
                   attack: AdversarialAttack,
                   device: torch.device = torch.device("cpu"),
                   verbose: bool = False
                   ) -> list[torch.Tensor]:
    try:
        model.eval()
        adversarial_images = []
        for images, labels in tqdm(data_loader, desc="Generating adversarial samples", disable=not verbose):
            images = images.to(device)
            labels = labels.to(device)
            # Generate adversarial images
            adv_images = attack.attack(model, images, labels)
            adversarial_images.append(adv_images)

        return adversarial_images
    except Exception as e:
        print("Error occurred while executing attack!")
        raise e
