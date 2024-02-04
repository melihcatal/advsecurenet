import torch
from tqdm.auto import tqdm

from advsecurenet.attacks.lots import LOTS
from advsecurenet.dataloader.data_loader_factory import DataLoaderFactory
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs import attack_configs
from advsecurenet.utils.adversarial_target_generator import \
    AdversarialTargetGenerator
from cli.utils.config import build_config
from cli.utils.data import get_custom_data


class CLILOTSAttack:
    """CLI wrapper for LOTS attack.

    Args:
        config_data (dict): The configuration data.

    """

    def __init__(self, config_data: dict, model: BaseModel, device: torch.device, dataset: torch.utils.data.TensorDataset):
        self.config_data = config_data
        self._validate_config()
        self.adversarial_target_generator = AdversarialTargetGenerator()
        self.model = model
        self.device = device
        self.dataset = dataset

    def execute_attack(self) -> list[torch.Tensor]:
        images = self.dataset.tensors[0]
        labels = self.dataset.tensors[1]

        # Generate target images
        target_images, target_labels = self._generate_target_images(
            data=images,
            labels=labels
        )

        # Adjust mode and configure attack
        self._adjust_mode()
        attack_config = build_config(
            self.config_data, attack_configs.LotsAttackConfig)
        attack = LOTS(attack_config)

        adversarial_images = self._perform_attack(
            attack, target_images, target_labels)

        return adversarial_images

    def _validate_config(self):
        if not self.config_data['deep_feature_layer']:
            raise ValueError(
                "Please provide deep feature layer for the attack!")
        if not self.config_data['auto_generate_target_images'] and not self.config_data['target_images_dir']:
            raise ValueError(
                "Please provide target images for the attack or set auto_generate_target_images to True!")

    def _generate_target_images(self, data, labels):
        if self.config_data['target_images_dir']:
            try:
                return get_custom_data(self.config_data['target_images_dir'])
            except Exception as e:
                raise ValueError(f"Error loading target images! Details: {e}")

        elif self.config_data['auto_generate_target_images']:
            paired = self.adversarial_target_generator.generate_target_images(
                zip(data, labels))

            _, _, target_images, target_labels = self.adversarial_target_generator.extract_images_and_labels(
                paired, data)
            return target_images, target_labels
        raise ValueError("Target image generation configuration not provided!")

    def _adjust_mode(self):
        """
        Sets the mode of the LOTS attack based on the configuration. It can be SINGLE or ITERATIVE.
        """
        mode_string = self.config_data.get("mode")
        self.config_data["mode"] = attack_configs.LotsAttackMode[mode_string.upper()]

    def _perform_attack(self, attack, target_images, target_labels):
        data_loader = DataLoaderFactory.create_dataloader(
            self.dataset, batch_size=self.config_data['batch_size'], shuffle=False)

        adversarial_images, total_samples = [], 0
        for images, labels in tqdm(data_loader, desc="Generating adversarial samples"):
            batch_size = images.size(0)
            target_images_batch, target_labels_batch = target_images[total_samples:total_samples +
                                                                     batch_size], target_labels[total_samples:total_samples + batch_size]

            images, labels, target_images_batch, target_labels_batch = [
                x.to(self.device) for x in [images, labels, target_images_batch, target_labels_batch]]
            adversarial_batch, _ = attack.attack(
                model=self.model, data=images, target=target_images_batch, target_classes=target_labels_batch)

            adversarial_images.append(adversarial_batch)
            total_samples += batch_size

        return adversarial_images
