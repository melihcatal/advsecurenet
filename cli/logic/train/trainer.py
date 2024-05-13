import os
from dataclasses import asdict
from typing import Optional

import click
import torch

from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs import TrainConfig
from advsecurenet.trainer.ddp_trainer import DDPTrainer
from advsecurenet.trainer.ddp_training_coordinator import \
    DDPTrainingCoordinator
from advsecurenet.trainer.trainer import Trainer
from cli.shared.types.train import TrainingCliConfigType
from cli.shared.utils.dataloader import get_dataloader
from cli.shared.utils.dataset import get_datasets
from cli.shared.utils.model import create_model


class CLITrainer:
    """
    Trainer class for the CLI. This module parses the CLI arguments and trains the model.

    Args:
        config (TrainingCliConfigType): The configuration for training.

    Attributes:
        config (TrainingCliConfigType): The configuration for training.

    """

    def __init__(self, config: TrainingCliConfigType):
        self.config: TrainingCliConfigType = config
        self.train_dataset = self._prepare_dataset()

    def train(self):
        """
        The main training function. This function parses the CLI arguments and executes the training.
        """

        try:
            if self.config.device.use_ddp:
                self._execute_ddp_training()
            else:
                self._execute_training()

        except Exception as e:
            raise e

    def _execute_ddp_training(self) -> None:
        """
        DDP Training function. Initializes the DDPTrainingCoordinator and runs the training.
        """
        # if no gpu ids are provided, use all available gpus
        if self.config.device.gpu_ids is None or len(self.config.device.gpu_ids) == 0:
            self.config.device.gpu_ids = list(range(torch.cuda.device_count()))

        world_size = len(self.config.device.gpu_ids)

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            [str(i) for i in self.config.device.gpu_ids])

        ddp_trainer = DDPTrainingCoordinator(
            self._ddp_training_fn,
            world_size,
        )

        if self.config.training.verbose:
            click.echo(
                f"Running DDP training on {world_size} GPUs with the following IDs: {self.config.device.gpu_ids}")

        ddp_trainer.run()

    def _ddp_training_fn(self, rank: int, world_size: int) -> None:
        train_config = self._prepare_training_environment(
            rank)

        ddp_trainer = DDPTrainer(train_config, rank, world_size)
        ddp_trainer.train()

    def _execute_training(self) -> None:
        train_config = self._prepare_training_environment()

        trainer = Trainer(train_config)
        trainer.train()

    def _prepare_training_environment(self, rank: Optional[int] = None) -> TrainConfig:
        """
        Prepare the common training environment components like model and dataloader.
        """
        if rank is not None:
            model = self._initialize_model(rank)
        else:
            model = self._initialize_model()

        train_loader = self._prepare_dataloader()
        train_config = self._prepare_train_config(model, train_loader)

        return train_config

    def _initialize_model(self, rank: Optional[int] = None) -> BaseModel:
        """
        Initialize the model.

        Args:
            rank (Optional[int]): The rank of the process in DDP training.

        Returns:
            BaseModel: The initialized model.
        """
        if self.config.training.verbose:
            if (rank is not None and rank == 0) or rank is None:
                click.echo(
                    f"""Loading model... with model name:
                    {self.config.model.model_name} and num_classes:
                    {self.config.model.num_classes} and num_input_channels:
                    {self.config.model.num_input_channels} """)

        return create_model(self.config.model)

    def _prepare_dataset(self):
        """
        Prepare the datasets.
        """
        train_data, _ = get_datasets(config=self.config.dataset)
        return train_data

    def _prepare_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Initialize the dataloader for single process training.

        Returns:      

            torch.utils.data.DataLoader: The training dataloader.      
        """
        train_data_loader = get_dataloader(
            config=self.config.dataloader,
            dataset=self.train_dataset,
            dataset_type='train',
            use_ddp=self.config.device.use_ddp
        )
        return train_data_loader

    def _prepare_train_config(self, model: BaseModel, train_data_loader: torch.utils.data.DataLoader) -> TrainConfig:
        """
        Prepare the training config.
        """
        config = TrainConfig(
            model=model, train_loader=train_data_loader, **asdict(self.config.training), **asdict(self.config.device))
        return config