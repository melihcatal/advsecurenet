
import os
import random
import click
import torch
from typing import cast, Type, Optional
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
from torch.utils.data import Subset
from advsecurenet.utils.adversarial_target_generator import AdversarialTargetGenerator
from advsecurenet.attacks.adversarial_attack import AdversarialAttack
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.attack_configs.attack_config import AttackConfig
from advsecurenet.shared.types.configs.configs import ConfigType
from advsecurenet.shared.types.dataset import DatasetType
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.datasets.dataset_factory import DatasetFactory
from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import AdversarialTrainingConfig
from advsecurenet.utils.model_utils import load_model
from advsecurenet.shared.types.attacks import AttackType as AdversarialAttackType
from advsecurenet.defenses.adversarial_training import AdversarialTraining
from advsecurenet.shared.types.configs import attack_configs
from advsecurenet.defenses.ddp_adversarial_training import DDPAdversarialTraining
from advsecurenet.utils.ddp_training_coordinator import DDPTrainingCoordinator
from cli.utils.config import build_config, load_configuration
from cli.types.adversarial_training import ATCliConfigType, AttackConfigDict, AttackWithConfigDict, ModelWithConfigDict


class AdversarialTrainingCLI:
    """
    Adversarial Training CLI class.

    This class is used to run adversarial training from the CLI.
    """

    def __init__(self, config_data: ATCliConfigType):
        self.config_data: ATCliConfigType = config_data
        self.config: AdversarialTrainingConfig
        self.adversarial_target_generator = AdversarialTargetGenerator()

    def train(self):
        """
        Public method to run adversarial training.
        """
        click.echo(click.style("Configuring adversarial training...",
                               fg="green"))
        dataset_name = self._validate_dataset_name()
        target_model = self._create_target_model()
        models = self._prepare_models()

        train_data, test_data = self._load_datasets(dataset_name)

        # use a subset of the train data if the number of samples is specified

        # TODO: use get_subset_data
        if self.config_data.num_samples_train:
            # randomly sample from the train data
            train_data = Subset(
                train_data, random.sample(range(len(train_data)), self.config_data.num_samples_train))

        if self.config_data.num_samples_test:
            # randomly sample from the test data
            test_data = Subset(
                test_data, random.sample(range(len(test_data)), self.config_data.num_samples_test))

        attacks = self._prepare_attacks()

        train_data_loader = self._setup_data_loaders(train_data)

        self.config = self._configure_adversarial_training(
            target_model, models, attacks, train_data_loader)

        if self.config.use_ddp:
            self._execute_ddp_adversarial_training(train_data)
        else:
            self._execute_adversarial_training(
                dataset_name, attacks)

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

    def _get_attacks(self) -> list[AttackWithConfigDict]:
        """
        Get the attack objects based on the attack names.

        Returns:
            list[AttackWithConfigDict]: List of attack objects.
        """
        attacks = []
        for attack_dict in self.config_data.attacks:
            for key, value in attack_dict.items():
                attack_name = key

                value = cast(AttackConfigDict, value)
                attack_config_path = value.get('config')
                config_data = load_configuration(
                    config_type=ConfigType.ATTACK,
                    config_file=str(attack_config_path),
                )

                attack_type = AdversarialAttackType[attack_name.upper()]
                attack_config_class = self._get_attack_config(attack_name)
                attack_config = build_config(config_data, attack_config_class)

                # if use_ddp is set to True, then set the distributed flag to True in case it is not already set
                if self.config_data.use_ddp:
                    attack_config.distributed_mode = True
                else:
                    # make sure the correct device setting is shared with the attack config
                    attack_config.device = self.config_data.device

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
        return [str(attack_name) for attack_name in self.config_data.attacks]

    def _create_target_model(self) -> BaseModel:
        """
        Create the target model.

        Returns:
            BaseModel: The target model.

        Raises:
            ValueError: If the model name is not supported.
        """
        return ModelFactory.create_model(
            self.config_data.model, num_classes=self.config_data.num_classes)

    def _prepare_models(self) -> list[BaseModel]:
        """
        Prepare the models that will be used to generate adversarial examples.

        Returns:
            list[BaseModel]: List of models.
        """
        if self.config_data.models is None:
            return [self._create_target_model()]

        models = [self._create_model(model_config)
                  for model_config in self.config_data.models]
        self._add_target_model(models)
        return models

    def _create_model(self, model_config: ModelWithConfigDict):
        for model_name, configs in model_config.items():
            return self._initialize_model(model_name, **configs)

    def _initialize_model(self, model_name: str, **configs) -> BaseModel:
        custom = configs.get('custom')
        pretrained = configs.get('pretrained')
        weights_path = configs.get('weights_path')
        num_classes = self.config_data.num_classes

        model = ModelFactory.create_model(
            model_name, num_classes=num_classes, pretrained=(not custom and pretrained))

        if custom and weights_path:
            model = load_model(model, weights_path)

        if custom and not weights_path:
            raise ValueError(
                "Custom model must have a weights path specified!")

        return model

    def _add_target_model(self, models: list[BaseModel]):
        """
        Add the target model to the list of models if it is not already present. Currently, this is done by checking if the model name is the same.

        Args:
            models (list[BaseModel]): List of models.
        """
        target_model = self._create_target_model()
        # Assuming each model has a unique 'model_name' attribute
        target_model_name = getattr(target_model, 'model_name', None)

        if target_model_name and not any(model.model_name == target_model_name for model in models):
            models.append(target_model)

    def _prepare_attacks(self) -> list[AttackWithConfigDict]:
        """
        Prepare the attacks.

        Returns:
            list[AttackWithConfigDict]: List of attacks with their configs.
        """
        return self._get_attacks()

    def _load_datasets(self, dataset_name) -> tuple[TorchDataset, TorchDataset]:
        """
        Load the dataset.

        Args:
            dataset_name (str): Name of the dataset.

        Returns:
            tuple[TorchDataset, TorchDataset]: Tuple of train and test datasets.
        """

        dataset_type = DatasetType(dataset_name)
        dataset_obj = DatasetFactory.create_dataset(dataset_type)
        train_data = dataset_obj.load_dataset(
            train=True,
            root=self.config_data.train_dataset_path)
        test_data = dataset_obj.load_dataset(
            train=False,
            root=self.config_data.test_dataset_path)

        return train_data, test_data

    def _setup_data_loaders(self, train_data: TorchDataset, test_data: Optional[TorchDataset] = None) -> tuple[TorchDataLoader, Optional[TorchDataLoader]]:
        """
        Setup the data loaders.

        Args:
            train_data (TorchDataset): The train dataset.
            test_data (TorchDataset): The test dataset.

        Returns:
            tuple[TorchDataLoader, TorchDataLoader]: Tuple of train and test data loaders.
        """
        train_data_loader = DataLoaderFactory.create_dataloader(
            train_data,
            batch_size=self.config_data.batch_size,
            shuffle=self.config_data.shuffle_train)
        if test_data is not None:
            test_data_loader = DataLoaderFactory.create_dataloader(
                test_data,
                batch_size=self.config_data.batch_size,
                shuffle=self.config_data.shuffle_test)
            return train_data_loader, test_data_loader

        return train_data_loader

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
            use_ddp=self.config_data.use_ddp,
            gpu_ids=self.config_data.gpu_ids,
            pin_memory=self.config_data.pin_memory,
        )

    def _execute_ddp_adversarial_training(self, train_data: TorchDataset) -> None:
        if self.config.gpu_ids is None:
            self.config.gpu_ids = list(range(torch.cuda.device_count()))

        world_size = len(self.config.gpu_ids)

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            str(x) for x in self.config.gpu_ids)

        click.echo(
            f"Running DDP adversarial training on {world_size} GPUs with the following IDs: {self.config.gpu_ids}")

        ddp_trainer = DDPTrainingCoordinator(
            self._ddp_training_fn,
            world_size,
            train_data,
        )
        ddp_trainer.run()

    def _ddp_training_fn(self, rank: int, world_size: int, train_data):
        """
        This function is used to run adversarial training using DDP.

        Args:
            rank (int): The rank of the current process.
            world_size (int): The number of processes to spawn.
        """
        train_data_loader = DataLoaderFactory.create_dataloader(
            train_data,
            batch_size=self.config_data.batch_size,
            shuffle=self.config_data.shuffle_train,
            pin_memory=self.config_data.pin_memory,
            sampler=DistributedSampler(train_data))

        self.config.train_loader = train_data_loader

        ddp_trainer = DDPAdversarialTraining(
            self.config, rank, world_size)
        ddp_trainer.train()

    def _execute_adversarial_training(self, dataset_name: str, attacks: list[AdversarialAttack]) -> None:
        """
        Execute adversarial training.

        Args:
            dataset_name (str): Name of the dataset.
            attacks (list[AdversarialAttack]): List of attacks.

        Raises:
            ValueError: If the dataset name is not supported.
        """
        attack_names_to_print = [
            attack.__class__.__name__ for attack in attacks]
        click.echo(click.style(
            f"Training on {dataset_name} with attacks {attack_names_to_print}...", fg="green"))
        adversarial_training = AdversarialTraining(self.config)
        adversarial_training.train()
        click.echo(click.style(
            f"Finished training on {dataset_name} with attacks {attack_names_to_print}!", fg="blue"))
