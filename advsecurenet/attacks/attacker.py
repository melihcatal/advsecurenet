import click
import torch
from tqdm.auto import tqdm

from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.evaluation.adversarial_evaluator import AdversarialEvaluator
from advsecurenet.shared.types.configs.attack_configs.attacker_config import \
    AttackerConfig
from cli.logic.attack.attacks.lots import CLILOTSAttack


class Attacker:
    """
    Attacker module is specialized module for attacking a model.
    """

    def __init__(self, config: AttackerConfig):
        self._config = config
        self._device = self._setup_device()
        self._model = self._setup_model()
        self._dataloader = self._create_dataloader()

    def execute(self):
        """
        Entry point for the attacker module. This function executes the attack.
        """
        return self._execute_attack()

    def _execute_attack(self):
        adversarial_images = []

        with AdversarialEvaluator(["attack_success_rate"]) as evaluator:
            data_iterator = self._get_iterator()

            self._model.eval()
            for images, labels in data_iterator:
                images, labels = self._prepare_data(images, labels)
                original_preds = self._get_predictions(images)
                adv_images = self._generate_adversarial_images(images, labels)

                evaluator.update(model=self._model,
                                 original_images=images,
                                 true_labels=labels,
                                 adversarial_images=adv_images)

                if torch.cuda.is_available():
                    # Free up memory
                    images = images.cpu()
                    labels = labels.cpu()
                    original_preds = original_preds.cpu()
                    adv_images = adv_images.cpu()
                    torch.cuda.empty_cache()

                if self._config.return_adversarial_images:
                    adversarial_images.append(adv_images)

            results = evaluator.get_results()
            self._summarize_results(results)

        return adversarial_images

    def _summarize_results(self, results: dict):
        """
        Summarizes the results of the attack.

        Args:
            results (dict): The results of the attack.
        """
        click.secho(
            f"Attack success rate: {results['attack_success_rate']:.2f}", fg="green")

    def _create_dataloader(self):
        return DataLoaderFactory.create_dataloader(self._config.dataloader)

    def _prepare_data(self, images, labels):
        images = images.to(self._device)
        labels = labels.to(self._device)
        return images, labels

    def _get_predictions(self, images):
        return torch.argmax(self._model(images), dim=1)

    def _generate_adversarial_images(self, images, labels):
        return self._config.attack.attack(self._model, images, labels)

    def _get_iterator(self):
        return tqdm(self._dataloader, leave=False, position=1, unit="batch", desc="Generating adversarial samples", colour="red")

    def _setup_device(self) -> torch.device:
        """
        Setup the device.
        """
        device = torch.device(
            self._config.device.processor) if self._config.device.processor else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        return device

    def _setup_model(self) -> torch.nn.Module:
        """
        Initializes the model and moves it to the device.
        """
        return self._config.model.to(self._device)
