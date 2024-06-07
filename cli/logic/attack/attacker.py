from typing import Union

import click
import torch
from torch.utils.data import Subset, random_split

from advsecurenet.attacks.attacker import Attacker, AttackerConfig
from advsecurenet.attacks.attacker.ddp_attacker import DDPAttacker
from advsecurenet.datasets.base_dataset import DatasetWrapper
from advsecurenet.datasets.targeted_adv_dataset import AdversarialDataset
from advsecurenet.shared.types.attacks import AttackType
from advsecurenet.shared.types.configs.dataloader_config import \
    DataLoaderConfig
from advsecurenet.utils.adversarial_target_generator import \
    AdversarialTargetGenerator
from cli.shared.types.attack import BaseAttackCLIConfigType
from cli.shared.types.utils.target import TargetCLIConfigType
from cli.shared.utils.dataset import get_datasets
from cli.shared.utils.helpers import read_data_from_file, save_images
from cli.shared.utils.model import create_model


class CLIAttacker:
    """
    Attacker class for the CLI. This module parses the CLI arguments and executes the attack.
    """

    def __init__(self, config: BaseAttackCLIConfigType, attack_type: AttackType):
        self._config: BaseAttackCLIConfigType = config
        self._attack_type: AttackType = attack_type
        self._adv_target_generator = AdversarialTargetGenerator()

    def execute(self):
        """
        The main attack function. This function parses the CLI arguments and executes the attack.
        """
        config = self._prepare_attack_config()
        if self._config.device.use_ddp:
            attacker = DDPAttacker(
                config=config,
                gpu_ids=self._config.device.gpu_ids)
        else:
            attacker = Attacker(config=config)

        adv_imgs = attacker.execute()

        if self._config.attack_procedure.save_result_images and adv_imgs:
            self._save_adversarial_images(adv_imgs)

        click.secho("Attack completed successfully.", fg="green")

    def _prepare_attack_config(self) -> AttackerConfig:
        """
        Prepare the attack configuration.

        Args:
            model (BaseModel): The model.
            data_loader (DataLoader): The data loader.

        Returns:
            AttackerConfig: The attack configuration.
        """

        attack = self._create_attack()
        model = create_model(self._config.model)
        dataloader_config = self._create_dataloader_config()

        config = AttackerConfig(
            model=model,
            attack=attack,
            dataloader=dataloader_config,
            device=self._config.device,
            return_adversarial_images=self._config.attack_procedure.save_result_images,
        )

        return config

    def _get_target_labels(self):
        if self._config.attack_config.target_parameters.targeted:
            target_labels = read_data_from_file(
                file_path=self._config.attack_config.target_parameters.target_labels_path,
                cast_type=int,
                return_type=list,
                separator=self._config.attack_config.target_parameters.target_labels_separator)
            return target_labels
        return None

    def _create_attack(self):
        attack_config = self._config.attack_config.attack_parameters
        try:
            attack_config.targeted = self._config.attack_config.target_parameters.targeted or False
        except:
            attack_config.targeted = False

        # Set the device for the attack. This is a workaround for now until we refactor the device handling
        attack_config.device = self._config.device

        attack_class = self._attack_type.value
        attack = attack_class(attack_config)
        return attack

    def _create_dataloader_config(self):
        dataset = self._prepare_dataset()
        dataloader_config = DataLoaderConfig(
            dataset=dataset,
            batch_size=self._config.dataloader.default.batch_size,
            num_workers=self._config.dataloader.default.num_workers,
            shuffle=self._config.dataloader.default.shuffle,
            drop_last=self._config.dataloader.default.drop_last,
            pin_memory=self._config.dataloader.default.pin_memory,
            sampler=None  # this will be set by the attacker later
        )
        return dataloader_config

    def _prepare_dataset(self) -> Union[torch.utils.data.TensorDataset, AdversarialDataset]:
        """
        Prepare the dataset.

        Returns:
            torch.utils.data.TensorDataset: The dataset.
        """
        target_parameters = self._get_target_parameters()
        if target_parameters and (target_parameters.target_labels_config.target_labels_path or target_parameters.target_labels_config.target_labels):
            target_labels = self._get_target_labels()

        train_data, test_data = get_datasets(
            self._config.dataset)

        # first check if the user wants to use only the train or test data
        if self._config.dataset.dataset_part == "train":
            all_data = self._validate_dataset_availability(
                train_data, "train")
        elif self._config.dataset.dataset_part == "test":
            all_data = self._validate_dataset_availability(
                test_data, "test")
        else:
            # if no dataset part is specified, use all the available data
            all_data = train_data + test_data if train_data and test_data else train_data or test_data

        # finally, sample the data if required
        if self._config.dataset.random_sample_size is not None and self._config.dataset.random_sample_size > 0:
            all_data = self._sample_data(
                all_data, self._config.dataset.random_sample_size)

        if target_parameters and target_parameters.targeted and target_parameters.auto_generate_target:
            all_data = self._generate_target(all_data)
        elif target_parameters and target_parameters.targeted and target_labels:
            all_data = AdversarialDataset(
                base_dataset=all_data,
                target_labels=target_labels)

            # if target_parameters and target_parameters.targeted and target_labels:
            #     # wrap the dataset with the adversarial dataset to include the target labels
            #     all_data = AdversarialDataset(
            #         base_dataset=all_data,
            #         target_labels=target_labels)

        return all_data

    def _generate_target(self, data: DatasetWrapper) -> AdversarialDataset:
        # if the attack is lots and targeted and auto_generate_target is set to True, generate target labels and target images
        if self._attack_type == AttackType.LOTS:

            paired = self._adv_target_generator.generate_target_images(
                train_data=data,
                total_tries=3)

            _, _, target_images, target_labels = self._adv_target_generator.extract_images_and_labels(
                paired=paired,
                dataset=data,
            )

            adv_data = AdversarialDataset(
                base_dataset=data,
                target_labels=target_labels,
                target_images=target_images
            )
        else:
            target_labels = self._adv_target_generator.generate_target_labels(
                data, overwrite=True)
            adv_data = AdversarialDataset(
                base_dataset=data,
                target_labels=target_labels)

        return adv_data

    def _get_target_parameters(self) -> Union[TargetCLIConfigType, None]:
        if hasattr(self._config.attack_config, "target_parameters"):
            return self._config.attack_config.target_parameters
        return None

    def _validate_dataset_availability(self, dataset, part):
        """Ensures that the specified dataset part is available."""
        if not dataset:
            raise ValueError(
                f"The dataset part '{part}' is specified but no {part} data is available. "
                "If you provide a path to a dataset, make sure to provide the {part} part."
            )
        return dataset

    def _sample_data(self, data, sample_size):
        """
        Sample data from the dataset.

        Args:
            data (torch.utils.data.Dataset): The dataset.
            sample_size (int): The sample size.

        Returns:
            torch.utils.data.Subset: The sampled data.
        """
        random_samples = min(sample_size, len(data))
        lengths = [random_samples, len(data) - random_samples]
        subset, _ = random_split(data, lengths)
        random_data = Subset(data, subset.indices)
        return random_data

    def _save_adversarial_images(self, adv_images):
        """
        Save the adversarial images.
        """
        save_images(images=adv_images,
                    path=self._config.attack_procedure.result_images_dir or "results",
                    prefix=self._config.attack_procedure.result_images_prefix or "adv"
                    )
