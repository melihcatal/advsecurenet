import click
import torch
from tqdm import tqdm

from advsecurenet.attacks.lots import LOTS
from advsecurenet.dataloader.data_loader_factory import DataLoaderFactory
from advsecurenet.shared.types.configs import attack_configs
from advsecurenet.utils.adversarial_target_generator import \
    AdversarialTargetGenerator
from cli.utils.config import build_config
from cli.utils.data import get_custom_data, load_and_prepare_data
from cli.utils.model import prepare_model


class CLILOTSAttack:
    """CLI wrapper for LOTS attack.

    Args:
        config_data (dict): The configuration data.

    """

    def __init__(self, config_data):
        self.config_data = config_data
        self._validate_config()
        self.adversarial_target_generator = AdversarialTargetGenerator()

    def execute_attack(self):
        print("Generating adversarial samples using LOTS attack...")
        dataset, num_classes, device = load_and_prepare_data(self.config_data)
        images = dataset.tensors[0]
        labels = dataset.tensors[1]

        # Generate target images
        target_images, target_labels = self._generate_target_images(
            data=images,
            labels=labels
        )

        # Adjust mode and configure attack
        self._adjust_mode()
        attack_config = build_config(
            self.config_data, attack_configs.LotsAttackConfig)
        model = prepare_model(self.config_data, num_classes, device)
        attack = LOTS(attack_config)

        return self._perform_attack(attack, model, dataset, device, target_images, target_labels)

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

            original_images, original_labels, target_images, target_labels = self.adversarial_target_generator.extract_images_and_labels(
                paired, data, "cuda:2")
            return target_images, target_labels
        raise ValueError("Target image generation configuration not provided!")

    def _adjust_mode(self):
        mode_string = self.config_data.get("mode")
        self.config_data["mode"] = attack_configs.LotsAttackMode[mode_string.upper()]

    def _perform_attack(self, attack, model, data, device, target_images, target_labels):
        data_loader = DataLoaderFactory.create_dataloader(
            data, batch_size=self.config_data['batch_size'], shuffle=False)

        adversarial_images, successful_attacks, total_samples = [], 0, 0
        for images, labels in tqdm(data_loader, desc="Generating adversarial samples"):
            batch_size = images.size(0)
            target_images_batch, target_labels_batch = target_images[total_samples:total_samples +
                                                                     batch_size], target_labels[total_samples:total_samples + batch_size]

            images, labels, target_images_batch, target_labels_batch = [
                x.to(device) for x in [images, labels, target_images_batch, target_labels_batch]]
            adversarial_batch, is_found = attack.attack(
                model=model, data=images, target=target_images_batch, target_classes=target_labels_batch)
            adversarial_preds = torch.argmax(model(adversarial_batch), dim=1)
            successful_attacks += (adversarial_preds ==
                                   target_labels_batch).sum().item()

            adversarial_images.append(adversarial_batch)
            total_samples += batch_size
            if self.config_data['verbose']:
                click.echo(
                    f"Attack success rate: {successful_attacks / total_samples * 100:.2f}%")

        success_rate = (successful_attacks / total_samples) * 100
        print(
            f"Succesfully generated adversarial samples! Attack success rate: {success_rate:.2f}%")
        return adversarial_images
