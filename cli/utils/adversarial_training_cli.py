
import os
import click
import pkg_resources
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
from typing import Dict, cast, Any, Type
from dataclasses import dataclass
from advsecurenet.attacks.adversarial_attack import AdversarialAttack
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.attack_configs.attack_config import AttackConfig
from advsecurenet.shared.types.configs.configs import ConfigType
from advsecurenet.shared.types.dataset import DatasetType
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.datasets.dataset_factory import DatasetFactory
from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import AdversarialTrainingConfig
from advsecurenet.shared.types.configs import TestConfig
from advsecurenet.utils.model_utils import save_model, load_model
from advsecurenet.shared.types.attacks import AttackType as AdversarialAttackType
from advsecurenet.defenses.adversarial_training import AdversarialTraining
from cli.utils.config import build_config, load_configuration
from cli.types.adversarial_training import ATCliConfigType, AttackConfigDict, AttackWithConfigDict
from advsecurenet.shared.types.configs import attack_configs


class AdversarialTrainingCLI:
    """
    Adversarial Training CLI class.

    This class is used to run adversarial training from the CLI.
    """

    def __init__(self, config_data: ATCliConfigType):
        self.config_data = config_data

    def _get_attack_config(self, attack_name: str) -> Type[AttackConfig]:
        """
        Get the attack config class based on the attack name.

        Args:
            attack_name (str): Name of the attack.

        Returns:
            Type[AttackConfig]: Attack config class.

        Examples:
            >>> self._get_attack_config("fgsm")
            <class 'advsecurenet.shared.types.configs.attack_configs.fgsm_attack_config.FgsmAttackConfig'>
        """

        attack_name = attack_name.lower()
        if attack_name == "fgsm":
            return attack_configs.FgsmAttackConfig
        elif attack_name == "pgd":
            return attack_configs.PgdAttackConfig
        elif attack_name == "cw":
            return attack_configs.CWAttackConfig
        elif attack_name == "deepfool":
            return attack_configs.DeepFoolAttackConfig
        elif attack_name == "lots":
            return attack_configs.LotsAttackConfig
        else:
            raise ValueError("Unsupported attack name!")

    def _get_attacks(self, config_attacks: list[AttackWithConfigDict]) -> list[AttackWithConfigDict]:
        """
        Get the attack objects based on the attack names.

        Args:
            config_attacks (list[AttackWithConfigDict]): List of attacks with their configs.

        Returns:
            list[AttackWithConfigDict]: List of attack objects.
        """
        attacks = []
        for attack_dict in config_attacks:
            for key, value in attack_dict.items():
                attack_name = key
                # This assumes that config file is the first element in the list
                # TODO: Fix this
                value = cast(list[AttackConfigDict], value)
                attack_config_path = value[0].get('config')
                config_data = load_configuration(
                    config_type=ConfigType.ATTACK,
                    config_file=str(attack_config_path),
                )
                attack_type = AdversarialAttackType[attack_name.upper()]
                attack_config_class = self._get_attack_config(attack_name)
                attack_config = build_config(config_data, attack_config_class)
                attack_class = attack_type.value
                attack = attack_class(attack_config)
                attacks.append(attack)
        return attacks

    def _validate_dataset_name(self) -> str:
        """
        Validate the dataset name.

        Returns:
            str: The validated dataset name.

        Raises:
            ValueError: If the dataset name is not supported.
        """
        dataset_name = self.config_data.dataset_type.upper()
        if dataset_name not in DatasetType._value2member_map_:
            raise ValueError("Unsupported dataset name! Choose from: " +
                             ", ".join([e.value for e in DatasetType]))
        return dataset_name

    def _extract_attack_names(self) -> list[str]:
        """
        Extract the attack names from the config data.

        Returns:
            list[str]: List of attack names.

        Examples:
            >>> self._extract_attack_names()
            ['fgsm', 'pgd']
        """
        return [attack_name for attack_name in self.config_data.attacks]

    def _create_target_model(self) -> BaseModel:
        """
        Create the target model.

        Returns:
            BaseModel: The target model.

        Raises:
            ValueError: If the model name is not supported.
        """
        return ModelFactory.get_model(
            self.config_data.model, num_classes=self.config_data.num_classes)

    def _prepare_models(self):
        """
        Prepare the models that will be used to generate adversarial examples.

        Returns:
            list[BaseModel]: List of models.
        """
        models = []
        for model_name in self.config_data.models:
            models.append(ModelFactory.get_model(
                model_name, num_classes=self.config_data.num_classes))

        target_model = self._create_target_model()
        if target_model not in models:
            models.append(target_model)
        return models

    def _prepare_attacks(self) -> list[AttackWithConfigDict]:
        """
        Prepare the attacks.

        Returns:
            list[AttackWithConfigDict]: List of attacks with their configs.
        """
        return self._get_attacks(self.config_data.attacks)

    def _load_datasets(self, dataset_name) -> tuple[TorchDataset, TorchDataset]:
        """
        Load the dataset.

        Args:
            dataset_name (str): Name of the dataset.

        Returns:
            tuple[TorchDataset, TorchDataset]: Tuple of train and test datasets.
        """

        dataset_type = DatasetType(dataset_name)
        dataset_obj = DatasetFactory.load_dataset(dataset_type)
        train_data = dataset_obj.load_dataset(train=True)
        test_data = dataset_obj.load_dataset(train=False)
        return train_data, test_data

    def _setup_data_loaders(self, train_data: TorchDataset, test_data: TorchDataset) -> tuple[TorchDataLoader, TorchDataLoader]:
        """
        Setup the data loaders.

        Args:
            train_data (TorchDataset): The train dataset.
            test_data (TorchDataset): The test dataset.

        Returns:
            tuple[TorchDataLoader, TorchDataLoader]: Tuple of train and test data loaders.
        """
        train_data_loader = DataLoaderFactory.get_dataloader(
            train_data, batch_size=self.config_data.batch_size, shuffle=True)
        test_data_loader = DataLoaderFactory.get_dataloader(
            test_data, batch_size=self.config_data.batch_size, shuffle=False)
        return train_data_loader, test_data_loader

    def _configure_adversarial_training(self, target_model: BaseModel, models: list[BaseModel], attacks: list[AdversarialAttack], train_data_loader: TorchDataLoader) -> AdversarialTrainingConfig:
        """
        Configure adversarial training.

        Args:
            target_model (BaseModel): The target model.
            models (list[BaseModel]): List of models.
            attacks (list[AdversarialAttack]): List of attacks.
            train_data_loader (TorchDataLoader): The train data loader.

        Returns:
            AdversarialTrainingConfig: The configuration for adversarial training.
        """
        return AdversarialTrainingConfig(
            model=target_model,
            models=models,
            attacks=attacks,
            train_loader=train_data_loader,
            epochs=self.config_data.epochs,
            learning_rate=self.config_data.learning_rate,
            save_checkpoint=self.config_data.save_checkpoint,
            save_checkpoint_path=self.config_data.save_checkpoint_path,
            save_checkpoint_name=self.config_data.save_checkpoint_name,
            load_checkpoint=self.config_data.load_checkpoint,
            load_checkpoint_path=self.config_data.load_checkpoint_path,
            verbose=self.config_data.verbose,
            adv_coeff=self.config_data.adv_coeff,
            optimizer=self.config_data.optimizer,
            criterion=self.config_data.criterion,
            device=self.config_data.device,
        )

    def _execute_adversarial_training(self, dataset_name: str, attacks: list[AdversarialAttack]) -> None:
        """
        Execute adversarial training.

        Args:
            dataset_name (str): Name of the dataset.
            attacks (list[AdversarialAttack]): List of attacks.

        Raises:
            ValueError: If the dataset name is not supported.
        """
        click.echo(
            f"Starting adversarial training on {dataset_name} with attacks {self.config_data.attacks} on target model {self.config_data.model}!")
        adversarial_training = AdversarialTraining(self.config)
        adversarial_training.train()
        click.echo(
            f"Model trained on {dataset_name} with attacks {attacks}!")

    def train(self):
        """
        Public method to run adversarial training.
        """
        dataset_name = self._validate_dataset_name()
        target_model = self._create_target_model()
        models = self._prepare_models()
        attacks = self._prepare_attacks()

        train_data, test_data = self._load_datasets(dataset_name)
        train_data_loader, test_data_loader = self._setup_data_loaders(
            train_data, test_data)

        self.config = self._configure_adversarial_training(
            target_model, models, attacks, train_data_loader)
        self._execute_adversarial_training(dataset_name, attacks)
