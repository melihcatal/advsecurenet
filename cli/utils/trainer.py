import os
from typing import Optional, Tuple

import click
import pkg_resources
import torch
from cli.types.training import TrainingCliConfigType
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler

from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.datasets.base_dataset import BaseDataset
from advsecurenet.datasets.dataset_factory import DatasetFactory
from advsecurenet.models.base_model import BaseModel
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types.configs import TrainConfig
from advsecurenet.shared.types.dataset import DatasetType
from advsecurenet.utils.ddp_trainer import DDPTrainer
from advsecurenet.utils.ddp_training_coordinator import DDPTrainingCoordinator
from advsecurenet.utils.model_utils import save_model
from advsecurenet.utils.trainer import Trainer


class CLITrainer:
    """
    Trainer class for the CLI. This module parses the CLI arguments and trains the model.
    """

    def __init__(self, config: TrainingCliConfigType):
        self._validate_config(config)
        self.config_data = config
        self.config: Optional[TrainConfig] = None
        self._initialize_params()

    def train(self):
        """
        The main training function. This function parses the CLI arguments and executes the training.
        """
        try:
            if self.config_data.use_ddp:
                self._execute_ddp_training()
            else:
                self._execute_training()

        except Exception as e:
            raise e

    def _initialize_params(self):
        """
        Initialize the parameters.
        """
        self.database_name = self._validate_dataset_name()
        self.train_data, self.test_data = self._load_datasets(
            self.database_name)

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

    def _validate_config(self, config):
        if not config.model_name or not config.dataset_name:
            raise ValueError(
                "Please provide both model name and dataset name!")

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
            train=True, root=self.config_data.train_dataset_path)
        test_data = dataset_obj.load_dataset(
            train=False, root=self.config_data.test_dataset_path)
        return train_data, test_data

    def _prepare_dataloader(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Initialize the dataloader for single process training.
        """
        train_data_loader = DataLoaderFactory.create_dataloader(
            self.train_data,
            batch_size=self.config_data.batch_size,
            shuffle=False if self.config_data.use_ddp else self.config_data.shuffle_train,
            num_workers=self.config_data.num_workers_train,
            drop_last=self.config_data.drop_last_train,
            pin_memory=self.config_data.pin_memory,
            sampler=DistributedSampler(
                self.train_data) if self.config_data.use_ddp else None
        )

        test_data_loader = DataLoaderFactory.create_dataloader(
            self.test_data,
            batch_size=self.config_data.batch_size,
            shuffle=self.config_data.shuffle_test,
            num_workers=self.config_data.num_workers_test,
            drop_last=self.config_data.drop_last_test,
            pin_memory=self.config_data.pin_memory)

        return train_data_loader, test_data_loader

    def _initialize_model(self, rank: Optional[int] = None) -> BaseModel:
        """
        Initialize the model.
        """
        if self.config_data.verbose:
            if (rank is not None and rank == 0) or rank is None:
                click.echo(
                    f"Loading model... with model name: {self.config_data.model_name} and num_classes: {self.config_data.num_classes} and num_input_channels: {self.config_data.num_input_channels} ")
        return ModelFactory.create_model(self.config_data.model_name, num_classes=self.config_data.num_classes, num_input_channels=self.config_data.num_input_channels)

    def _prepare_train_config(self, model: BaseModel, train_data_loader: torch.utils.data.DataLoader) -> TrainConfig:
        """
        Prepare the training config.
        """
        return TrainConfig(
            model=model,
            train_loader=train_data_loader,
            epochs=self.config_data.epochs,
            learning_rate=self.config_data.lr,
            use_ddp=self.config_data.use_ddp,
            device=self.config_data.device,
            save_final_model=self.config_data.save_final_model,
            save_model_path=self.config_data.save_model_path if self.config_data.save_model_path else os.getcwd(),
            save_model_name=self.config_data.save_model_name if self.config_data.save_model_name else self.config_data.model_name + "_final",
            save_checkpoint=self.config_data.save_checkpoint,
            save_checkpoint_path=self.config_data.save_checkpoint_path if self.config_data.save_checkpoint_path else os.getcwd(),
            save_checkpoint_name=self.config_data.save_checkpoint_name if self.config_data.save_checkpoint_name else self.config_data.model_name + "_checkpoint",
            checkpoint_interval=self.config_data.checkpoint_interval,
            load_checkpoint=self.config_data.load_checkpoint,
            load_checkpoint_path=self.config_data.load_checkpoint_path if self.config_data.load_checkpoint_path else os.getcwd(),
            scheduler=self.config_data.scheduler,
            scheduler_kwargs=self.config_data.scheduler_kwargs,
            optimizer=self.config_data.optimizer,
            optimizer_kwargs=self.config_data.optimizer_kwargs,
            criterion=self.config_data.loss
        )

    def _execute_ddp_training(self) -> None:
        """
        DDP Training function. Initializes the DDPTrainingCoordinator and runs the training.
        """
        # if no gpu ids are provided, use all available gpus
        if self.config_data.gpu_ids is None or len(self.config_data.gpu_ids) == 0:
            self.config_data.gpu_ids = list(range(torch.cuda.device_count()))

        world_size = len(self.config_data.gpu_ids)

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            str(x) for x in self.config_data.gpu_ids)

        ddp_trainer = DDPTrainingCoordinator(
            self._ddp_training_fn,
            world_size,
        )

        if self.config_data.verbose:
            click.echo(
                f"Running DDP training on {world_size} GPUs with the following IDs: {self.config_data.gpu_ids}")

        ddp_trainer.run()

    def _ddp_training_fn(self, rank: int, world_size: int) -> None:
        """
        The main training function for DDP. This function is called by each process spawned by the DDPTrainingCoordinator.
        """
        # the model must be initialized in each process
        model = self._initialize_model(rank)

        train_loader = self._prepare_dataloader()[0]

        train_config = self._prepare_train_config(model, train_loader)

        ddp_trainer = DDPTrainer(
            train_config, rank, world_size)

        ddp_trainer.train()

    def _execute_training(self) -> None:
        """
        Single process training function. Initializes the Trainer and runs the training.
        """

        train_data_loader = self._prepare_dataloader()[0]

        model = self._initialize_model()

        train_config = self._prepare_train_config(
            model, train_data_loader)

        trainer = Trainer(train_config)
        trainer.train()
