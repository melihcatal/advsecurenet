import os
import click
import pkg_resources
import torch
from typing import Tuple
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset as TorchDataset
from cli.types.training import TrainingCliConfigType
from advsecurenet.shared.types.dataset import DatasetType
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.datasets.dataset_factory import DatasetFactory
from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.shared.types.configs import TrainConfig
from advsecurenet.utils.trainer import Trainer
from advsecurenet.utils.ddp_training_coordinator import DDPTrainingCoordinator
from advsecurenet.utils.ddp_trainer import DDPTrainer
from advsecurenet.datasets.base_dataset import BaseDataset
from advsecurenet.models.base_model import BaseModel


class CLITrainer:
    """
    Trainer class for the CLI. This module parses the CLI arguments and trains the model.
    """

    def __init__(self, config: TrainingCliConfigType):
        self._validate_config(config)
        self.config_data = config
        self.config: TrainConfig = None

    def train(self):
        try:
            self._set_save_path()
            dataset_name = self._validate_dataset_name()
            train_data, test_data, dataset_obj = self._load_datasets(
                dataset_name)
            train_data_loader, test_data_loader = self._prepare_dataloader(
                train_data, test_data)
            model = self._initialize_model()
            self.config = self._prepare_train_config(
                model, train_data_loader, dataset_obj)
            if self.config_data.use_ddp:
                self._execute_ddp_training(train_data)
            else:
                self._execute_training()
            if self.config.save_final_model:
                self._save_trained_model(
                    model, self.config_data.dataset_name.upper())
            click.echo(
                f"Model trained on {self.config_data.dataset_name.upper()}!")
        except Exception as e:
            raise e

    def _validate_dataset_name(self) -> str:
        """
        Validate the dataset name.

        Returns:
            str: The validated dataset name.

        Raises:
            ValueError: If the dataset name is not supported.
        """
        dataset_name = self.config_data.dataset_name.upper()
        if dataset_name not in DatasetType._value2member_map_:
            raise ValueError("Unsupported dataset name! Choose from: " +
                             ", ".join([e.value for e in DatasetType]))
        return dataset_name

    def _set_save_path(self):
        if not self.config_data.save_path:
            self.config_data.save_path = pkg_resources.resource_filename(
                "advsecurenet", "weights")

    def _validate_config(self, config):
        if not config.model_name or not config.dataset_name:
            raise ValueError(
                "Please provide both model name and dataset name!")

    def _load_datasets(self, dataset_name) -> tuple[TorchDataset, TorchDataset, BaseDataset]:
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
            train=True, root=self.config_data.train_dataset_path)
        test_data = dataset_obj.load_dataset(
            train=False, root=self.config_data.test_dataset_path)
        return train_data, test_data, dataset_obj

    def _prepare_dataloader(self, train_data: TorchDataset, test_data: TorchDataset) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Initialize the dataloader for single process training.
        """
        train_data_loader = DataLoaderFactory.create_dataloader(
            train_data,
            batch_size=self.config_data.batch_size,
            shuffle=self.config_data.shuffle_train,
            num_workers=self.config_data.num_workers_train,
            drop_last=self.config_data.drop_last_train,
            pin_memory=self.config_data.pin_memory)

        test_data_loader = DataLoaderFactory.create_dataloader(
            test_data,
            batch_size=self.config_data.batch_size,
            shuffle=self.config_data.shuffle_test,
            num_workers=self.config_data.num_workers_test,
            drop_last=self.config_data.drop_last_test,
            pin_memory=self.config_data.pin_memory)

        return train_data_loader, test_data_loader

    def _initialize_model(self) -> BaseModel:
        return ModelFactory.create_model(self.config_data.model_name, num_classes=self.config_data.num_classes)

    def _prepare_train_config(self, model: BaseModel, train_data_loader: torch.utils.data.DataLoader, dataset_obj: BaseDataset) -> TrainConfig:
        """
        Prepare the training config.
        """
        return TrainConfig(
            model=model,
            train_loader=train_data_loader,
            epochs=self.config_data.epochs,
            learning_rate=self.config_data.lr,
            use_ddp=self.config_data.use_ddp,
            device=self.config_data.device
        )

    def _execute_training(self) -> None:
        """
        Single process training function. Initializes the Trainer and runs the training.
        """
        trainer = Trainer(self.config)
        trainer.train()

    def _execute_ddp_training(self, train_data: BaseDataset) -> None:
        """
        DDP Training function. Initializes the DDPTrainingCoordinator and runs the training.
        """
        if self.config.gpu_ids is None:
            self.config.gpu_ids = list(range(torch.cuda.device_count()))

        world_size = len(self.config_data.gpu_ids)

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            str(x) for x in self.config_data.gpu_ids)

        ddp_trainer = DDPTrainingCoordinator(
            self._ddp_training_fn,
            world_size,
            train_data,
        )

        if self.config_data.verbose:
            click.echo(
                f"Running DDP training on {world_size} GPUs with the following IDs: {self.config_data.gpu_ids}")

        ddp_trainer.run()

    def _ddp_training_fn(self, rank: int, world_size: int, train_data: BaseDataset) -> None:
        """
        The main training function for DDP. This function is called by each process spawned by the DDPTrainingCoordinator.
        """

        # Need to override the train loader for
        train_data_loader = DataLoaderFactory.create_dataloader(
            train_data, batch_size=self.config_data.batch_size,
            shuffle=self.config_data.shuffle_train,
            pin_memory=self.config_data.pin_memory,
            sampler=DistributedSampler(train_data),
            num_workers=self.config_data.num_workers_train,
        )

        self.config.train_loader = train_data_loader

        ddp_trainer = DDPTrainer(
            self.config, rank, world_size)
        ddp_trainer.train()
