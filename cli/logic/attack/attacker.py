import logging
from typing import Optional, Union

import click
import torch
from torch.utils.data import Subset, random_split

from advsecurenet.attacks.attacker import Attacker, AttackerConfig
from advsecurenet.attacks.attacker.ddp_attacker import DDPAttacker
from advsecurenet.datasets.base_dataset import DatasetWrapper
from advsecurenet.datasets.targeted_adv_dataset import AdversarialDataset
from advsecurenet.distributed.ddp_coordinator import DDPCoordinator
from advsecurenet.shared.types.attacks import AttackType
from advsecurenet.shared.types.configs.dataloader_config import \
    DataLoaderConfig
from advsecurenet.utils.adversarial_target_generator import \
    AdversarialTargetGenerator
from advsecurenet.utils.ddp import set_visible_gpus
from cli.shared.types.attack import BaseAttackCLIConfigType
from cli.shared.types.utils.target import TargetCLIConfigType
from cli.shared.utils.dataset import get_datasets
from cli.shared.utils.helpers import read_data_from_file, save_images
from cli.shared.utils.model import create_model

logger = logging.getLogger(__name__)


class CLIAttacker:
    """
    Attacker class for the CLI. This module parses the CLI arguments and executes the attack.
    """

    def __init__(self, config: BaseAttackCLIConfigType, attack_type: AttackType, **kwargs):
        self._config: BaseAttackCLIConfigType = config
        self._attack_type: AttackType = attack_type
        self._adv_target_generator = AdversarialTargetGenerator()
        self._kwargs = kwargs
        self._dataset = self._prepare_dataset()

    def execute(self):
        """
        The main attack function. This function parses the CLI arguments and executes the attack.
        """
        logger.info("Starting %s attack.", self._attack_type.value)
        if self._config.device.use_ddp:
            logger.info("Using DDP for attack with the following GPUs: %s",
                        self._config.device.gpu_ids)
            self._execute_ddp_attack()

        else:
            logger.info("Using single GPU or CPU for attack.")
            self._execute_attack()

        click.secho("Attack completed successfully.", fg="green")
        logger.info("%s attack completed successfully.",
                    self._attack_type.value)

    def _execute_attack(self):
        """ 
        Non-Distributed attack function. Initializes the attacker and runs the attack.
        """
        config = self._prepare_attack_config()
        attacker = Attacker(config=config, **self._kwargs)
        adv_imgs = attacker.execute()
        self._save_images_if_needed(adv_imgs)

    def _execute_ddp_attack(self) -> None:
        """
        DDP Training function. Initializes the DDPCoordinator and runs the training.
        """
        if self._config.device.gpu_ids is None or len(self._config.device.gpu_ids) == 0:
            self._config.device.gpu_ids = list(
                range(torch.cuda.device_count()))

        world_size = len(self._config.device.gpu_ids)
        set_visible_gpus(self._config.device.gpu_ids)

        ddp_attacker = DDPCoordinator(self._ddp_attack_fn, world_size)
        ddp_attacker.run()

        if self._config.attack_procedure.save_result_images:
            gathered_adv_images = DDPAttacker.gather_results(world_size)
            self._save_adversarial_images(gathered_adv_images)

    def _ddp_attack_fn(self, rank: int, world_size: int) -> None:
        attack_config = self._prepare_attack_config()
        ddp_attacker = DDPAttacker(attack_config, rank, world_size)
        ddp_attacker.execute()

    def _save_images_if_needed(self,
                               adv_imgs: Optional[list] = None,
                               world_size: Optional[int] = None
                               ):
        """ 
        Save the adversarial images if needed. If the attack is distributed, gather the results from all processes.

        Args:
            adv_imgs (Optional[list]): The adversarial images.
            world_size (Optional[int]): The total number of processes. Needed for distributed attacks. If not provided and the attack is distributed, the function will gather the results from the configured world size.

        """

        if self._config.attack_procedure.save_result_images and self._config.device.use_ddp and (world_size or len(self._config.device.gpu_ids)) and not adv_imgs:
            logger.info("Gathering results from all processes.")
            adv_imgs = DDPAttacker.gather_results(world_size)

        if adv_imgs:
            logger.info("Saving adversarial images.")
            self._save_adversarial_images(adv_imgs)
            logger.info("Adversarial images saved successfully.")
        else:
            logger.info("No adversarial images to save.")

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
            evaluators=self._kwargs.get("evaluators", ["attack_success_rate"])
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
        except AttributeError:
            attack_config.targeted = False

        # Set the device for the attack. This is a workaround for now until we refactor the device handling
        attack_config.device = self._config.device

        attack_class = self._attack_type.value
        attack = attack_class(attack_config)
        return attack

    def _create_dataloader_config(self):
        dataloader_config = DataLoaderConfig(
            dataset=self._dataset,
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
        target_labels = self._get_target_labels_if_available(target_parameters)

        train_data, test_data = get_datasets(self._config.dataset)

        all_data = self._select_data_partition(train_data, test_data)

        all_data = self._sample_data_if_required(all_data)

        all_data = self._generate_or_assign_target_labels(
            all_data, target_parameters, target_labels)

        return all_data

    def _get_target_labels_if_available(self, target_parameters):
        """
        Check if target labels are available and return them if they are.

        """
        if target_parameters and (target_parameters.target_labels_config.target_labels_path or target_parameters.target_labels_config.target_labels):
            return self._get_target_labels()
        return None

    def _select_data_partition(self, train_data, test_data):
        dataset_part = self._config.dataset.dataset_part
        if dataset_part == "train":
            return self._validate_dataset_availability(train_data, "train")
        elif dataset_part == "test":
            return self._validate_dataset_availability(test_data, "test")
        else:
            return train_data + test_data if train_data and test_data else train_data or test_data

    def _sample_data_if_required(self, all_data):
        sample_size = self._config.dataset.random_sample_size
        if sample_size is not None and sample_size > 0:
            return self._sample_data(all_data, sample_size)
        return all_data

    def _generate_or_assign_target_labels(self, all_data, target_parameters, target_labels):
        if target_parameters and target_parameters.targeted:
            if target_parameters.auto_generate_target:
                logger.info("Generating target labels and images.")
                all_data = self._generate_target(all_data)
                logger.info(
                    "Target labels and images generated successfully. Total length of the dataset: %s", len(all_data))
            elif target_labels:
                all_data = AdversarialDataset(
                    base_dataset=all_data, target_labels=target_labels)
                logger.info(
                    "Target labels are provided. Total length of the dataset: %s", len(all_data))
        return all_data

    def _generate_target(self, data: DatasetWrapper) -> AdversarialDataset:
        # if the attack is lots and targeted and auto_generate_target is set to True, generate target labels and target images
        if self._attack_type == AttackType.LOTS:
            target_images, target_labels = self._adv_target_generator.generate_target_images_and_labels(
                data=data)
            logger.info("Successfully extracted target labels and images.")

            adv_data = AdversarialDataset(
                base_dataset=data,
                target_labels=target_labels,
                target_images=target_images
            )
        else:
            target_labels = self._adv_target_generator.generate_target_labels(
                data=data, overwrite=True)
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
