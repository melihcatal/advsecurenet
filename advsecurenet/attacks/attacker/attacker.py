import logging
from typing import Optional, Union

import click
import torch
from tqdm.auto import tqdm

from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.evaluation.adversarial_evaluator import AdversarialEvaluator
from advsecurenet.shared.types.configs.attack_configs.attacker_config import \
    AttackerConfig

logger = logging.getLogger(__name__)


class Attacker:
    """
    Attacker module is specialized module for attacking a model.
    """

    def __init__(self, config: AttackerConfig,  **kwargs):
        self._config = config
        self._device = self._setup_device()
        self._model = self._setup_model()
        self._dataloader = self._create_dataloader()
        self._kwargs = kwargs

    def execute(self):
        """
        Entry point for the attacker module. This function executes the attack.
        """
        return self._execute_attack()

    def _setup_device(self) -> torch.device:
        """
        Setup the device.
        """
        if self._config.device.processor:
            device = torch.device(self._config.device.processor)
        else:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        return device

    def _create_dataloader(self):
        """
        It is possible to pass a DataLoader object directly to the AttackerConfig. If not, we create a DataLoader object from the DataLoaderConfig.
        """
        if isinstance(self._config.dataloader, torch.utils.data.DataLoader):
            return self._config.dataloader
        return DataLoaderFactory.create_dataloader(self._config.dataloader)

    def _setup_model(self) -> torch.nn.Module:
        """
        Initializes the model and moves it to the device.
        """
        return self._config.model.to(self._device)

    def _execute_attack(self) -> Union[None, list[torch.Tensor]]:
        """
        Executes the attack and returns the adversarial images if required.
        """
        adversarial_images = []

        with AdversarialEvaluator(evaluators=self._config.evaluators,
                                    target_models=self._kwargs.get(
                                        "target_models", [])
                                    ) as evaluator:
            data_iterator = self._get_iterator()

            self._model.eval()
            for data in data_iterator:
                if self._config.attack.targeted and len(data) == 4:
                    # Dataset returns (images, true_labels, target_images, target_labels) i.e. LOTS
                    images, true_labels, target_images, target_labels = data
                else:
                    # Dataset returns (images, labels)
                    images, true_labels = data
                    target_labels = true_labels
                    target_images = None

                images, true_labels, target_labels = self._prepare_data(
                    images, true_labels, target_labels)

                adv_images = self._generate_adversarial_images(
                    images,
                    target_labels if self._config.attack.targeted else true_labels,
                    target_images
                )
                evaluator.update(model=self._model,
                                    original_images=images,
                                    true_labels=true_labels,
                                    adversarial_images=adv_images,
                                    is_targeted=self._config.attack.targeted,
                                    target_labels=target_labels)

                if torch.cuda.is_available() and self._device.type == "cuda":
                    # Free up memory
                    images = images.cpu()
                    true_labels = true_labels.cpu()
                    target_labels = target_labels.cpu()
                    adv_images = adv_images.cpu()
                    with torch.cuda.device(self._device):
                        torch.cuda.empty_cache()

                if self._config.return_adversarial_images:
                    adversarial_images.append(adv_images)

            results = evaluator.get_results()
            self._summarize_results(results)

        return adversarial_images if self._config.return_adversarial_images else None

    def _prepare_data(self, *args):
        """
        Move the required data to the device.
        """
        return [arg.to(self._device) for arg in args]

    def _get_predictions(self, images):
        return torch.argmax(self._model(images), dim=1)

    def _generate_adversarial_images(self,
                                     images: torch.Tensor,
                                     labels: torch.Tensor,
                                     target_images: Optional[torch.Tensor] = None
                                     ):
        """
        Running the attack to generate adversarial images.

        Args:
            images (torch.Tensor): The input images.
            labels (torch.Tensor): The true labels.
            target_images (Optional[torch.Tensor]): The target images for the attack. This is only used for certain attacks i.e. LOTS.
        """
        return self._config.attack.attack(self._model, images, labels, target_images)

    def _summarize_results(self, results: dict) -> None:
        """
        Summarizes the results of the attack.

        Args:
            results (dict): The results of the attack.
        """
        logger.info("Results summary: %s", results)

        for metric_name, metric_value in results.items():
            if isinstance(metric_value, dict):
                for sub_metric_name, sub_metric_value in metric_value.items():
                    full_metric_name = f"{metric_name}-{sub_metric_name}"
                    self._summarize_metric(full_metric_name, sub_metric_value)
            else:
                self._summarize_metric(metric_name, metric_value)

    def _summarize_metric(self, name, value):
        local_results = torch.tensor(value, device=self._device)
        click.secho(
            f"{name.replace('_', ' ').title()}: {local_results.item():.4f}", fg='green')

    def _get_iterator(self):
        return tqdm(self._dataloader, leave=False, position=1, unit="batch", desc="Generating adversarial samples", colour="red")
