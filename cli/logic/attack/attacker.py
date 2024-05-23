

import click
import torch
from torch.utils.data import Subset, random_split

from advsecurenet.attacks.attacker import Attacker, AttackerConfig
from advsecurenet.attacks.ddp_attacker import DDPAttacker
from advsecurenet.shared.types.attacks import AttackType
from advsecurenet.shared.types.configs.dataloader_config import \
    DataLoaderConfig
from advsecurenet.utils.dataclass import filter_for_dataclass
from cli.shared.types.attack import BaseAttackCLIConfigType
from cli.shared.types.utils.dataset import DatasetCliConfigType
from cli.shared.utils.dataset import get_datasets
from cli.shared.utils.helpers import save_images
from cli.shared.utils.model import create_model


class CLIAttacker:
    """
    Attacker class for the CLI. This module parses the CLI arguments and executes the attack.
    """

    def __init__(self, config: BaseAttackCLIConfigType, attack_type: AttackType):
        self._config: BaseAttackCLIConfigType = config
        self._attack_type: AttackType = attack_type

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
            return_adversarial_images=self._config.attack_procedure.save_result_images
        )

        return config

    def _create_attack(self):
        attack_config = self._config.attack_config

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

    def _prepare_dataset(self) -> torch.utils.data.TensorDataset:
        """
        Prepare the dataset.

        Returns:
            torch.utils.data.TensorDataset: The dataset.
        """
        dataset_cfg = self._create_dataset_config()

        train_data, test_data = get_datasets(dataset_cfg)

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

        return all_data

    def _create_dataset_config(self) -> DatasetCliConfigType:
        dataset_cfg = filter_for_dataclass(
            data=self._config.dataset,
            dataclass_type=DatasetCliConfigType,
            convert=True
        )
        return dataset_cfg

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
