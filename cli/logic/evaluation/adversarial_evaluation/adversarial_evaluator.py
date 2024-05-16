import json
import os
from typing import List

import click
import torch
from tqdm.auto import tqdm

from advsecurenet.attacks.adversarial_attack import AdversarialAttack
from advsecurenet.evaluation.adversarial_evaluator import AdversarialEvaluator
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.attacks import \
    AttackType as AdversarialAttackType
from advsecurenet.shared.types.configs.attack_configs.attack_config import \
    AttackConfig
from advsecurenet.shared.types.configs.configs import ConfigType
from cli.shared.types.evaluation import AdversarialEvaluationCliConfigType
from cli.shared.types.utils.model import ModelCliConfigType
from cli.shared.utils.attack_mappings import attack_cli_mapping, attack_mapping
from cli.shared.utils.config import load_and_instantiate_config
from cli.shared.utils.data import load_and_prepare_data
from cli.shared.utils.dataloader import get_dataloader
from cli.shared.utils.model import create_model


class CLIAdversarialEvaluator():
    """
    Base class for the adversarial evaluation CLI.
    """

    def __init__(self, config: AdversarialEvaluationCliConfigType, **kwargs):
        self.config = config
        self.kwargs = kwargs

    def run(self):
        """
        Run the evaluation.
        """
        model, data_loader, attack, target_models = self._prepare_evaluation_env()
        self._execute_evaluation(
            model, data_loader, attack, target_models)

    def _prepare_attack(self) -> list[AdversarialAttack]:
        """
        Get the attack objects based on the attack names.

        Returns:
            list[AttackWithConfigDict]: List of attack objects.
        """
        attack_name = self.config.evaluation.attack.name.upper()
        attack_config_path = self.config.evaluation.attack.config

        attack_config: AttackConfig = load_and_instantiate_config(
            config=attack_config_path,
            default_config_file=f"{attack_name.lower()}_attack_base_config.yml",
            config_type=ConfigType.ATTACK,
            config_class=attack_mapping[attack_name],
        )

        attack_config.device = self.config.device
        attack_type, _ = attack_cli_mapping[attack_name]

        # if the attack is LOTS, we need to use the custom lots wrapper

        if attack_type == AdversarialAttackType.LOTS:
            # lots = CLILOTSAttack(config=attack_config,
            #                     dataset=dataset,
            #                     data_loader=data_loader,
            #                     model=model)
            pass
        else:
            attack_class = attack_type.value
            attack = attack_class(attack_config)

        return attack

    def _prepare_target_models(self) -> List[BaseModel]:
        """ 
        Prepare the models that will be used as the target models for the transferability evaluation.

        Returns:
            List[BaseModel]: List of initialized target models based on the configuration provided.

        """

        if not self.config.evaluation.target_models or not any(self.config.evaluation.target_models):
            return []

        # Load and initialize each model based on its configuration
        models = [
            create_model(
                load_and_instantiate_config(
                    config=model_config.get('config'),
                    default_config_file="model_config.yml",
                    config_type=ConfigType.MODEL,
                    config_class=ModelCliConfigType
                )
            )
            for model_config in self.config.evaluation.target_models
        ]

        return models

    def _prepare_evaluation_env(self) -> tuple[BaseModel, torch.utils.data.DataLoader, AdversarialAttack, List[BaseModel]]:
        """
        Prepares the environment for evaluation by loading the model, dataset, dataloader, attack, and target models.

        Returns:
            tuple[BaseModel, torch.utils.data.DataLoader, AdversarialAttack, List[BaseModel]]: The model, dataloader, attack, target models
        """
        model = create_model(self.config.model)
        dataset = load_and_prepare_data(self.config.dataset)
        data_loader = get_dataloader(
            config=self.config.dataloader,
            dataset=dataset,
            dataset_type='default',
            use_ddp=self.config.device.use_ddp)

        attack = self._prepare_attack()
        target_models = self._prepare_target_models()
        return model, data_loader, attack, target_models

    def _execute_evaluation(self,
                            model: BaseModel,
                            dataloader: torch.utils.data.DataLoader,
                            attack: AdversarialAttack,
                            target_models: List[BaseModel],
                            ):
        """
        Main method to execute the evaluation of the adversarial examples.

        Args:
            model (BaseModel): The model to evaluate.
            dataloader (torch.utils.data.DataLoader): The dataloader for the dataset.
            attack (AdversarialAttack): The attack to evaluate.
            target_models (List[BaseModel]): The list of target models to evaluate.


        """
        try:
            with AdversarialEvaluator(evaluators=self.config.evaluation.evaluators,
                                      target_models=target_models,
                                      ) as evaluator:
                for images, labels in tqdm(dataloader, total=len(dataloader), desc="Evaluating adversarial examples"):
                    adv_images = attack.attack(model=model, x=images, y=labels)
                    evaluator.update(model=model, original_images=images, true_labels=labels,
                                     adversarial_images=adv_images, is_targeted=False)

            click.secho("Evaluation completed successfully.", fg="green")
            click.secho("Results:", fg="green", bold=True)
            results = evaluator.get_results()
            self._print_results(results)

            if self.config.evaluation.save_results:
                self._save_results(results)

        except Exception as e:
            click.secho("Evaluation failed.", fg="red")
            click.secho(f"Error: {e}", fg="red")
            raise e

    def _print_results(self, results: dict):
        """
        Print the results of the evaluation in a readable JSON format.

        Args:
            results (dict): The results of the evaluation.
        """
        formatted_results = json.dumps(results, indent=4, sort_keys=True)
        for line in formatted_results.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                click.secho(key + ':', fg='yellow', nl=False, bold=True)
                click.secho(value, fg='white')
            else:
                click.secho(line, fg='cyan', bold=True)

    def _save_results(self, results: dict):
        """
        Save the results of the evaluation to a file.

        Args:
            results (dict): The results of the evaluation.
        """
        # if the output path is not provided use the current directory
        path = self.config.evaluation.save_path if self.config.evaluation.save_path else os.getcwd()
        os.makedirs(path, exist_ok=True)

        # add file name results.json to the path
        file_name = self.config.evaluation.save_filename if self.config.evaluation.save_filename else 'results.json'
        path = os.path.join(path, file_name)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, sort_keys=True)
            click.secho(f"Results saved to {path}", fg='green')
